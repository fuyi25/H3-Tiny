"""Packed MiniMax-H3 inputs with ground-truth targets, built by the official diffusers encoders.

Pruning calibration and OKD both need the transformer's exact inputs for a sample with a *known* target: the
conditioning rows, text embeddings and packed layout of an ordinary request, plus the ground-truth video and audio
latents in place of the generated rows. We get them from the official `MiniMaxH3Blocks` workflow (``ref2va`` for the
Ref2VA transformer; ``fl2va`` or ``t2va`` for the FL2VA transformer, which serves both) with its denoising loop
swapped for a capture step:

1. The ground-truth clip goes through the same normalization and encoders a reference video does (24 fps, LANCZOS
   onto the canvas, ImageNet-normalized VAE posterior sample, fp16 rounding; audio resampled to the audio VAE rate,
   posterior mode), and is passed in as the request's ``latents`` / ``audio_latents``.
2. The workflow encodes the prompt and conditions, builds the layout and packs everything, exactly as at inference.
   ``ref2va`` conditions on the sample's references; ``fl2va`` on the first and last frame of the ground-truth clip
   itself (after the same normalization), so the keyframes match the target exactly; ``t2va`` on text alone.
3. :class:`CaptureStep` takes the packed state instead of denoising it; nothing is decoded.

`prepare` is the expensive part (the 32B text encoder); its result is a small dict that is cached to disk and reused.
`noised_inputs` then puts the generated rows at any ``(t_video, t_audio)`` and returns the transformer's kwargs with
the flow-matching velocity target ``x0 - noise``.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

#: Fields of the packed layout the transformer reads by name.
LAYOUT_FIELDS = ("token_tags", "position_ids", "video_indices", "audio_indices", "text_indices")


WORKFLOWS = ("ref2va", "fl2va", "t2va")


@dataclass
class Sample:
    """One calibration / training sample: prompt, conditions and the ground-truth clip.

    ``references`` (in prompt order) are used by ``ref2va`` only; ``fl2va`` takes its keyframes from ``video``.
    """

    sample_id: str
    prompt: str
    video: str
    num_frames: int
    references: list[str] = field(default_factory=list)
    height: int | None = None
    width: int | None = None
    workflow: str = "ref2va"


def _capture_step_class():
    from diffusers.modular_pipelines import InputParam, ModularPipelineBlocks, OutputParam

    class CaptureStep(ModularPipelineBlocks):
        """Replaces the denoising loop: hands the packed transformer inputs out instead of denoising them."""

        model_name = "minimax-h3"

        @property
        def description(self) -> str:
            return "H3 Tiny: capture the packed transformer inputs (ground-truth target rows) for calibration / KD."

        @property
        def inputs(self) -> list:
            return [
                InputParam(name="latents", type_hint=torch.Tensor, required=True),
                InputParam(name="audio_latents", type_hint=torch.Tensor, required=True),
                InputParam.template("prompt_embeds"),
                InputParam(name="num_condition_video_rows", type_hint=int, required=True),
                InputParam(name="num_condition_audio_rows", type_hint=int, required=True),
                InputParam(kwargs_type="denoiser_input_fields"),
            ]

        @property
        def intermediate_outputs(self) -> list:
            return [OutputParam("prepared", type_hint=dict, description="Packed inputs with ground-truth targets.")]

        @torch.no_grad()
        def __call__(self, components, state):
            block_state = self.get_block_state(state)
            fields = {k: v.detach().cpu() for k, v in block_state.denoiser_input_fields.items() if k in LAYOUT_FIELDS}
            block_state.prepared = dict(
                latents=block_state.latents.detach().float().cpu(),
                audio_latents=block_state.audio_latents.detach().float().cpu(),
                prompt_embeds=block_state.prompt_embeds.detach().cpu(),
                num_condition_video_rows=int(block_state.num_condition_video_rows),
                num_condition_audio_rows=int(block_state.num_condition_audio_rows),
                keyframe_noise_aug=float(components.keyframe_noise_aug),
                **fields,
            )
            self.set_block_state(state, block_state)
            return components, state

    return CaptureStep


def capture_blocks(workflow: str = "ref2va"):
    """The official ``workflow`` blocks with the denoising loop replaced by `CaptureStep` and no decoding.

    ``get_workflow`` flattens the workflow into dotted top-level names (``denoise.set_timesteps``,
    ``denoise.denoise``, ``decode.video``, ...); the loop is the block that holds the loop denoiser.
    """
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Blocks
    from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3LoopDenoiser

    if workflow not in WORKFLOWS:
        raise ValueError(f"workflow must be one of {WORKFLOWS}, got {workflow!r}")
    blocks = MiniMaxH3Blocks().get_workflow(workflow)
    capture = _capture_step_class()()
    subs = blocks.sub_blocks
    loops = [name for name, block in subs.items()
             if any(isinstance(x, MiniMaxH3LoopDenoiser) for x in (getattr(block, "sub_blocks", None) or {}).values())]
    if len(loops) != 1:
        raise RuntimeError(f"expected exactly one {workflow} denoising loop, found {loops}")
    subs[loops[0]] = capture
    for name in [n for n in subs if n.endswith("after_denoise") or n.split(".")[0] == "decode"]:
        subs.pop(name)
    return blocks


class H3Batcher:
    """Encodes samples of one ``workflow`` into packed inputs with ground-truth targets (no transformer is loaded)."""

    def __init__(self, base_model: str, device: torch.device | str = "cuda", dtype: torch.dtype = torch.bfloat16,
                 workflow: str = "ref2va"):
        self.device = torch.device(device)
        self.workflow = workflow
        self.pipe = capture_blocks(workflow).init_pipeline(base_model)
        names = [n for n in self.pipe.pretrained_component_names if not n.startswith("transformer")]
        kwargs = dict(dtype=dtype)
        if Path(base_model).is_dir():
            kwargs["pretrained_model_name_or_path"] = base_model
        self.pipe.load_components(names=names, **kwargs)
        self.pipe.to(self.device)
        # The ground truth is encoded with the video and audio VAEs. t2va / fl2va use them only to decode (t2va has
        # no keyframe to encode), and the capture blocks drop decoding, so then take both from the ref2va blocks.
        self.encoders = self.pipe
        if any(getattr(self.pipe, n, None) is None for n in ("vae", "audio_vae")):
            self.encoders = capture_blocks("ref2va").init_pipeline(base_model)
            self.encoders.load_components(names=["vae", "audio_vae"], **kwargs)
            self.encoders.to(self.device)

    @torch.no_grad()
    def encode_target(self, sample: Sample, height: int | None = None, width: int | None = None
                      ) -> tuple[torch.Tensor, torch.Tensor, Any]:
        """Ground-truth ``(1, C, F, H, W)`` video and ``(2, C_a, N)`` audio latents, normalized like conditioning,
        and the normalized ``(F, H, W, 3)`` uint8 frames they were encoded from.

        The clip is fitted onto the official canvas for its own aspect; ``height`` / ``width``, when given, must match.
        """
        from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3VideoReference
        from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep
        from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
        from diffusers.modular_pipelines.minimax_h3.modular_pipeline import audio_latent_num_frames

        c = self.encoders
        clip = MiniMaxH3VideoReference.from_file(sample.video)
        frames = MiniMaxH3Ref2VASetupStep._normalize_video_condition(
            clip.frames, clip.fps, sample.num_frames, c.canvas_multiple, c.config.canvas_short_edge,
            c.config.canvas_max_pixels, c.fps)
        height, width = height or frames.shape[1], width or frames.shape[2]
        if frames.shape[0] != sample.num_frames or frames.shape[1:3] != (height, width):
            raise ValueError(f"{sample.sample_id}: ground truth gives {frames.shape[:3]}, layout needs "
                             f"({sample.num_frames}, {height}, {width}); the clip may be too short")
        pixels = torch.from_numpy(frames.copy()).to(self.device).permute(3, 0, 1, 2)[None]
        video = encode_vae_condition(c.vae, pixels, c.pixel_mean, c.pixel_std, c.keyframe_encode_seed)

        if clip.audio is None:
            raise ValueError(f"{sample.sample_id}: the ground-truth clip has no soundtrack")
        waveform = MiniMaxH3Ref2VASetupStep._normalize_audio_condition(
            clip.audio, clip.sample_rate, c.audio_sampling_rate, sample.num_frames / c.fps)
        audio_vae = c.audio_vae
        posterior = audio_vae.encode(waveform.to(self.device)[:, None], return_dict=False)[0]
        mean = torch.tensor(audio_vae.config.latents_mean).view(1, -1, 1)
        std = torch.tensor(audio_vae.config.latents_std).view(1, -1, 1)
        audio = (posterior.mode().float().cpu() - mean) / std  # (2, C_a, N)
        n = audio_latent_num_frames(sample.num_frames)
        if audio.shape[-1] < n:
            audio = torch.nn.functional.pad(audio, (0, n - audio.shape[-1]))
        return video, audio[..., :n].contiguous(), frames

    @torch.no_grad()
    def prepare(self, sample: Sample, seed: int = 1234) -> dict[str, Any]:
        """Run the official pre-denoising workflow with the ground truth as the generated rows."""
        from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference, MiniMaxH3VideoReference
        from PIL import Image

        if sample.workflow != self.workflow:
            raise ValueError(f"{sample.sample_id}: a {sample.workflow} sample given to a {self.workflow} batcher")
        video, audio, frames = self.encode_target(sample, sample.height, sample.width)
        # The generation canvas is the ground truth's, latent size times the VAE spatial compression.
        height, width = (n * self.encoders.vae_spatial_compression_ratio for n in video.shape[-2:])
        if self.workflow == "ref2va":
            conditions = dict(references=[
                (MiniMaxH3VideoReference if Path(p).suffix.lower() in (".mp4", ".mov", ".webm", ".mkv")
                 else MiniMaxH3ImageReference).from_file(p) for p in sample.references])
        elif self.workflow == "fl2va":
            conditions = dict(image=Image.fromarray(frames[0]), last_image=Image.fromarray(frames[-1]))
        else:
            conditions = {}
        prepared = self.pipe(prompt=sample.prompt, num_frames=sample.num_frames, height=height, width=width,
                             latents=video, audio_latents=audio, generator=torch.Generator().manual_seed(seed),
                             output="prepared", **conditions)
        prepared.update(sample_id=sample.sample_id, num_frames=sample.num_frames, height=height, width=width,
                        workflow=self.workflow)
        return prepared


def noised_inputs(prepared: dict[str, Any], t_video: float, t_audio: float, generator: torch.Generator,
                  device: torch.device | str = "cuda") -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Transformer kwargs with the generated rows at ``(t_video, t_audio)`` (``t = 1 - sigma``), and their targets.

    Conditioning rows keep the noise-augmentation level the workflow gave them, at the timestep inference uses for
    them; reference audio rows are clean (t = 1). Targets are the flow-matching velocities ``x0 - noise`` of the
    generated rows, in the order of the transformer's video / audio outputs after the conditioning rows.
    """
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3SetTimestepsStep

    ncv, nca = prepared["num_condition_video_rows"], prepared["num_condition_audio_rows"]
    video, audio = prepared["latents"].clone(), prepared["audio_latents"].clone()
    x0_v, x0_a = video[ncv:], audio[nca:]
    noise_v = torch.randn(x0_v.shape, generator=generator, dtype=torch.float32)
    noise_a = torch.randn(x0_a.shape, generator=generator, dtype=torch.float32)
    video[ncv:] = t_video * x0_v + (1 - t_video) * noise_v
    audio[nca:] = t_audio * x0_a + (1 - t_audio) * noise_a
    timestep, timestep_indices = MiniMaxH3SetTimestepsStep.build_row_timesteps(
        prepared["video_indices"], prepared["audio_indices"], ncv, nca, prepared["text_indices"].numel(),
        float(t_video), float(t_audio), max(float(t_video), prepared["keyframe_noise_aug"]), 1.0)
    kwargs = dict(hidden_states=video[None].to(device), audio_hidden_states=audio[None].to(device),
                  encoder_hidden_states=prepared["prompt_embeds"].to(device), timestep=timestep.to(device),
                  timestep_indices=timestep_indices.to(device),
                  **{k: prepared[k].to(device) for k in LAYOUT_FIELDS})
    targets = dict(video=(x0_v - noise_v), audio=(x0_a - noise_a))
    return kwargs, targets


def forward_velocity(transformer, kwargs: dict[str, Any], prepared: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the transformer and return the predicted velocity of the generated video / audio rows only."""
    params = inspect.signature(transformer.forward).parameters
    video, audio = transformer(**{k: v for k, v in kwargs.items() if k in params}, return_dict=False)
    return video[0, prepared["num_condition_video_rows"]:].float(), audio[0, prepared["num_condition_audio_rows"]:].float()
