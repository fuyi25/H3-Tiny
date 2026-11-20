"""H3 Tiny inference on the official diffusers MiniMax-H3 workflows.

Everything but the transformer and the step schedule is the official pipeline (text encoder, reference encoding,
packed layout, Euler update, VAE decode). Two blocks are replaced:

* the timestep step samples ``N`` model evaluations on ``sigma = linspace(1, 0, N + 1)`` shifted per modality (12
  video, 3 audio), the grid H3 Tiny was trained and validated on, and gives every row a ``(t, r)`` pair whose
  endpoint is the next grid point, ``r_i = 1 - sigma_{i+1}`` (conditioning rows stay pinned, ``r = t``);
* the loop denoiser passes ``endpoint_timestep`` to the transformer.

Full-step students (``endpoint_mode="none"``) ignore ``r`` and take ordinary Euler steps (50 is the default). MeanFlow
students (``endpoint_mode="meanflow"``) predict the average velocity over ``[t, r]`` and jump straight to the next grid
point, so they sample in few steps (`scripts/infer.py` uses 6 by default, 4 or more work).
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Any

import torch

from .modeling import load_h3tiny, stored_dtypes, build_h3tiny
from .prune.adaln import AUDIO_SHIFT, VIDEO_SHIFT

log = logging.getLogger(__name__)

#: Short names accepted by ``attention_backend``; anything else goes to diffusers' ``set_attention_backend`` as is.
ATTENTION_BACKENDS = {"sdpa": "native", "fa3": "_flash_3_hub"}

#: GPU memory kept free for activations under ``offload=True`` (a 10 s 768p clip needs about this much).
OFFLOAD_MARGIN = "24GB"

#: Transformer component of each official workflow.
TRANSFORMER_COMPONENTS = {"t2va": "transformer", "fl2va": "transformer", "ref2va": "transformer_ref"}


def shifted_sigmas(steps: int, shift: float) -> torch.Tensor:
    """``steps + 1`` sigmas: ``linspace(1, 0, steps + 1)`` through ``shift * s / (1 + (shift - 1) * s)``."""
    grid = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)
    return (shift * grid / (1.0 + (shift - 1.0) * grid)).to(torch.float32)


def row_time_pairs(video_indices, audio_indices, num_condition_video_rows: int, num_condition_audio_rows: int,
                   num_text_tokens: int, video_time: tuple[float, float], audio_time: tuple[float, float],
                   condition_video_timestep: float, condition_audio_timestep: float = 1.0):
    """The official per-row timestep assignment with an endpoint per row; unique as ``(t, r)`` pairs.

    Generated video rows (and text rows, which never reach a head) take ``video_time``, generated audio rows
    ``audio_time``; visual conditioning rows are held at their noise-augmentation level and reference audio at clean,
    both with ``r = t``. Returns ``(timestep, endpoint_timestep, timestep_indices)``.
    """
    video_indices, audio_indices = video_indices.cpu(), audio_indices.cpu()
    n = int(video_indices.numel() + audio_indices.numel() + num_text_tokens)
    pairs = torch.tensor(video_time, dtype=torch.float32).repeat(n, 1)
    pairs[video_indices[:num_condition_video_rows]] = float(condition_video_timestep)
    pairs[audio_indices[num_condition_audio_rows:]] = torch.tensor(audio_time, dtype=torch.float32)
    pairs[audio_indices[:num_condition_audio_rows]] = float(condition_audio_timestep)
    unique, indices = torch.unique(pairs, dim=0, sorted=True, return_inverse=True)
    return unique[:, 0].contiguous(), unique[:, 1].contiguous(), indices


def _block_classes():
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3SetTimestepsStep
    from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3LoopDenoiser

    class H3TinySetTimestepsStep(MiniMaxH3SetTimestepsStep):
        @property
        def description(self) -> str:
            return ("H3 Tiny: N evaluations on linspace(1, 0, N + 1) shifted 12 (video) / 3 (audio); stages one "
                    "`(timestep, endpoint_timestep, timestep_indices)` triple per step, endpoint = next grid point.")

        def __call__(self, components, state):
            block_state = self.get_block_state(state)
            steps = int(block_state.num_inference_steps)
            device = components._execution_device
            video_sigmas, audio_sigmas = shifted_sigmas(steps, VIDEO_SHIFT), shifted_sigmas(steps, AUDIO_SHIFT)
            components.scheduler.set_timesteps(sigmas=video_sigmas, device=device)
            components.audio_scheduler.set_timesteps(sigmas=audio_sigmas, device=device)
            block_state.timesteps = components.scheduler.timesteps
            block_state.audio_timesteps = components.audio_scheduler.timesteps
            # Read t back from the schedulers so the plan matches the update step exactly.
            video_t, audio_t = block_state.timesteps.cpu(), block_state.audio_timesteps.cpu()
            video_r, audio_r = 1.0 - video_sigmas[1:], 1.0 - audio_sigmas[1:]
            block_state.row_timestep_plan = [
                tuple(x.to(device) for x in row_time_pairs(
                    block_state.video_indices, block_state.audio_indices, block_state.num_condition_video_rows,
                    block_state.num_condition_audio_rows, block_state.text_indices.numel(),
                    (float(video_t[i]), float(video_r[i])), (float(audio_t[i]), float(audio_r[i])),
                    max(float(video_t[i]), components.keyframe_noise_aug)))
                for i in range(steps)
            ]
            self.set_block_state(state, block_state)
            return components, state

    class H3TinyLoopDenoiser(MiniMaxH3LoopDenoiser):
        @property
        def description(self) -> str:
            return "H3 Tiny: one forward per step, with each row's endpoint `r` passed as `endpoint_timestep`."

        @torch.no_grad()
        def __call__(self, components, block_state, i: int, t: torch.Tensor):
            transformer = getattr(components, self.transformer_name)
            timestep, endpoint_timestep, timestep_indices = block_state.row_timestep_plan[i]
            params = inspect.signature(transformer.forward).parameters
            layout = {k: v for k, v in block_state.denoiser_input_fields.items() if k in params}
            block_state.noise_pred, block_state.audio_noise_pred = transformer(
                hidden_states=block_state.latents[None], audio_hidden_states=block_state.audio_latents[None],
                encoder_hidden_states=block_state.prompt_embeds, timestep=timestep,
                endpoint_timestep=endpoint_timestep, timestep_indices=timestep_indices,
                attention_kwargs=block_state.attention_kwargs, return_dict=False, **layout)
            return components, block_state

    return MiniMaxH3SetTimestepsStep, MiniMaxH3LoopDenoiser, H3TinySetTimestepsStep, H3TinyLoopDenoiser


def h3tiny_blocks(workflow: str = "ref2va"):
    """The official ``workflow`` blocks with the H3 Tiny timestep step and loop denoiser."""
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Blocks

    base_steps, base_denoiser, steps_cls, denoiser_cls = _block_classes()
    blocks = MiniMaxH3Blocks().get_workflow(workflow)
    counts = {"steps": 0, "denoiser": 0}

    def swap(container) -> None:
        for name, block in list((getattr(container, "sub_blocks", None) or {}).items()):
            if isinstance(block, base_steps):
                container.sub_blocks[name] = steps_cls()
                counts["steps"] += 1
            elif isinstance(block, base_denoiser):
                container.sub_blocks[name] = denoiser_cls(transformer_name=block.transformer_name)
                counts["denoiser"] += 1
            else:
                swap(block)

    swap(blocks)
    if counts != {"steps": 1, "denoiser": 1}:
        raise RuntimeError(f"expected one timestep step and one loop denoiser in {workflow}, found {counts}")
    return blocks


def detect_endpoint_mode(weights: str | Path) -> str:
    """``"meanflow"`` if the checkpoint carries the MeanFlow interval embedder (``r_time_embedder``), else ``"none"``."""
    _, files = build_h3tiny(weights, endpoint_mode="none")
    return "meanflow" if any(k.startswith("r_time_embedder.") for k in stored_dtypes(files)) else "none"


def resolve_attention(backend: str | None, device: torch.device) -> str | None:
    """Diffusers attention backend for ``"auto"``, ``"sdpa"``, ``"fa3"`` or a diffusers backend name.

    ``"auto"`` picks FlashAttention-3 (from the Hub via ``pip install kernels``) on Hopper GPUs and PyTorch SDPA
    elsewhere; ``None`` means SDPA.
    """
    has_kernels = importlib.util.find_spec("kernels") is not None
    if backend in (None, "auto"):
        hopper = device.type == "cuda" and torch.cuda.get_device_capability(device)[0] == 9
        return "_flash_3_hub" if hopper and has_kernels else None
    if backend in ("fa3", "_flash_3_hub") and not has_kernels:
        log.warning("attention %s needs `pip install kernels`; falling back to sdpa", backend)
        return None
    return ATTENTION_BACKENDS.get(backend, backend)


def load_pipeline(weights: str | Path, base_model: str | Path, *, workflow: str = "ref2va",
                  endpoint_mode: str | None = None, device: str | torch.device = "cuda", offload: bool = False,
                  attention_backend: str | None = "auto") -> Any:
    """An official MiniMax-H3 ``workflow`` pipeline driving the H3 Tiny transformer at ``weights``.

    ``base_model`` provides the text encoder, VAEs, tokenizer and processors. ``offload=True`` keeps components on
    the CPU and moves each to ``device`` when it runs (needed below ~110 GB of GPU memory; the text encoder alone is
    ~64 GB in bf16). ``attention_backend`` is ``"auto"`` (FlashAttention-3 on Hopper when ``kernels`` is installed,
    else SDPA), ``"sdpa"``, ``"fa3"`` or any diffusers backend name; it applies to the transformer only. Call the result as ``pipe(prompt=..., references=[...], num_frames=..., height=..., width=...,
    num_inference_steps=..., generator=..., output=["videos", "audio", "sampling_rate"])``.
    """
    from diffusers.modular_pipelines import ComponentsManager

    device = torch.device(device)
    endpoint_mode = endpoint_mode or detect_endpoint_mode(weights)
    manager = ComponentsManager() if offload else None
    pipe = h3tiny_blocks(workflow).init_pipeline(str(base_model), components_manager=manager)
    component = TRANSFORMER_COMPONENTS[workflow]
    names = [n for n in pipe.pretrained_component_names if n != component]
    kwargs: dict[str, Any] = dict(dtype=torch.bfloat16)
    if Path(base_model).is_dir():
        kwargs["pretrained_model_name_or_path"] = str(base_model)
    pipe.load_components(names=names, **kwargs)
    missing = [n for n in names if getattr(pipe, n, None) is None]
    if missing:
        raise RuntimeError(f"MiniMax-H3 components did not load: {missing}")
    transformer = load_h3tiny(weights, endpoint_mode=endpoint_mode, device="cpu")
    backend = resolve_attention(attention_backend, device)
    if backend not in (None, "native"):
        transformer.set_attention_backend(backend)
        # set_attention_backend also moves diffusers' global default; keep the fp32 audio VAE on SDPA.
        if hasattr(getattr(pipe, "audio_vae", None), "set_attention_backend"):
            pipe.audio_vae.set_attention_backend("native")
    log.info("H3 Tiny %s on %s: attention %s, offload %s", workflow, device, backend or "sdpa", offload)
    pipe.update_components(**{component: transformer})
    if offload:
        # Keep room for activations: with the default 3 GB margin a GPU that can hold the text encoder and the
        # transformer together (e.g. the 33B on a 141 GB H200) keeps both resident and runs out in the first step.
        manager.enable_auto_cpu_offload(device=device, memory_reserve_margin=OFFLOAD_MARGIN)
    else:
        pipe.to(device)
    return pipe
