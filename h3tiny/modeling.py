"""Pruned MiniMax-H3 students (H3 Tiny) as a diffusers `MiniMaxH3Transformer3DModel` subclass.

Differences from the official transformer, all driven by `config.json`:

* pruned depth / FFN width: plain config values (`num_layers`, `ffn_dim`; the token refiner keeps
  `refiner_ffn_dim`);
* low-rank AdaLN (`adaln_rank`): `adaln_proj.linear_2(adaln_proj.linear_1(silu(temb)))` instead of
  `adaln_proj.linear`;
* optional interval branch (`r_time_embedder`) for students distilled with endpoint conditioning:
  `temb(t, r) = emb_t(t) + emb_r(r - t)` on the `t = 1 - sigma` axis. `endpoint_mode="r_equals_t"` pins r to t
  (velocity-KD students), `"none"` drops the branch (plain pruning exports).

Weights load strictly at their stored dtypes (bf16 blocks, fp32 embedders and heads).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import torch
from diffusers import MiniMaxH3Transformer3DModel
from diffusers.configuration_utils import register_to_config
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.models.modeling_utils import get_parameter_dtype
from diffusers.models.transformers.transformer_minimax_h3 import (
    MINIMAX_H3_MODALITY_NUM,
    MiniMaxH3TransformerOutput,
)
from safetensors import safe_open
from torch import Tensor, nn

ENDPOINT_MODES = ("meanflow", "r_equals_t", "none")


class LowRankModulation(nn.Module):
    """AdaLN modulation through a rank bottleneck: two biased linears instead of one 2688 -> 96768 projection."""

    def __init__(self, time_embed_dim: int, hidden_size: int, rank: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear_1 = nn.Linear(time_embed_dim, rank, bias=True)
        self.linear_2 = nn.Linear(rank, 6 * hidden_size * MINIMAX_H3_MODALITY_NUM, bias=True)

    def forward(self, temb: Tensor) -> tuple[Tensor, ...]:
        temb = self.linear_2(self.linear_1(nn.functional.silu(temb).to(get_parameter_dtype(self.linear_1))))
        return temb.view(-1, 6 * self.hidden_size).chunk(6, dim=-1)


class H3TinyTransformer3DModel(MiniMaxH3Transformer3DModel):
    _keep_in_fp32_modules: ClassVar[list[str]] = [*MiniMaxH3Transformer3DModel._keep_in_fp32_modules, "r_time_embedder"]

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 56,
        attention_head_dim: int = 128,
        hidden_size: int = 5376,
        num_layers: int = 50,
        num_refiner_layers: int = 2,
        ffn_dim: int = 14336,
        refiner_ffn_dim: int | None = None,
        in_channels: int = 24,
        audio_in_channels: int = 32,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        text_dim: int = 5120,
        freq_dim: int = 256,
        time_embed_hidden_dim: int = 5376,
        time_embed_dim: int = 2688,
        rope_freq_dim: int = 16,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
        final_norm_eps: float = 1e-5,
        adaln_rank: int | None = None,
        endpoint_mode: str = "meanflow",
    ) -> None:
        if endpoint_mode not in ENDPOINT_MODES:
            raise ValueError(f"endpoint_mode must be one of {ENDPOINT_MODES}, got {endpoint_mode!r}")
        # The token refiner keeps the official width when the block FFN is pruned; build it at that width.
        super().__init__(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_refiner_layers=num_refiner_layers,
            ffn_dim=refiner_ffn_dim or ffn_dim,
            in_channels=in_channels,
            audio_in_channels=audio_in_channels,
            patch_size=patch_size,
            text_dim=text_dim,
            freq_dim=freq_dim,
            time_embed_hidden_dim=time_embed_hidden_dim,
            time_embed_dim=time_embed_dim,
            rope_freq_dim=rope_freq_dim,
            rope_theta=rope_theta,
            norm_eps=norm_eps,
            qk_norm_eps=qk_norm_eps,
            final_norm_eps=final_norm_eps,
        )
        # The parent's `register_to_config` recorded the refiner width as `ffn_dim`; restore the block width.
        self.register_to_config(ffn_dim=ffn_dim, refiner_ffn_dim=refiner_ffn_dim)
        if (refiner_ffn_dim or ffn_dim) != ffn_dim:
            # Rebuild the block FFNs at the pruned width; the refiner above keeps its own.
            from diffusers.models.attention import FeedForward

            for block in self.transformer_blocks:
                block.ff = FeedForward(hidden_size, inner_dim=ffn_dim, activation_fn="swiglu", bias=False)
        if adaln_rank is not None:
            for block in self.transformer_blocks:
                block.adaln_proj = LowRankModulation(time_embed_dim, hidden_size, adaln_rank)
        if endpoint_mode != "none":
            self.r_time_embedder = TimestepEmbedding(
                in_channels=freq_dim, time_embed_dim=time_embed_hidden_dim, out_dim=time_embed_dim
            )

    def time_embedding(self, timestep: Tensor, endpoint_timestep: Tensor | None) -> Tensor:
        temb = self.time_embedder(self.time_proj(timestep).to(get_parameter_dtype(self.time_embedder)))
        mode = self.config.endpoint_mode
        if mode == "none":
            return temb
        if mode == "r_equals_t" or endpoint_timestep is None:
            endpoint_timestep = timestep
        if endpoint_timestep.shape != timestep.shape:
            raise ValueError(f"endpoint_timestep {list(endpoint_timestep.shape)} != timestep {list(timestep.shape)}")
        if bool((endpoint_timestep < timestep).any()):
            raise ValueError("MeanFlow endpoints must be dataward of their timesteps (r >= t)")
        interval = endpoint_timestep - timestep
        dtype = get_parameter_dtype(self.r_time_embedder)
        return temb + self.r_time_embedder(self.time_proj(interval).to(dtype))

    def forward(
        self,
        hidden_states: Tensor,
        audio_hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        timestep_indices: Tensor,
        token_tags: Tensor,
        position_ids: Tensor,
        video_indices: Tensor,
        audio_indices: Tensor,
        text_indices: Tensor,
        endpoint_timestep: Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> MiniMaxH3TransformerOutput | tuple[Tensor, Tensor]:
        sequence_length = position_ids.shape[0]
        if token_tags.shape != (sequence_length,) or timestep_indices.shape != (sequence_length,):
            raise ValueError("`token_tags` and `timestep_indices` must both be `(seq_len,)` tensors.")
        rotary_emb = self.rope(position_ids)
        video_embeds = self.proj_in(hidden_states.to(get_parameter_dtype(self.proj_in)))
        audio_embeds = self.audio_proj_in(audio_hidden_states.to(get_parameter_dtype(self.audio_proj_in)))
        text_embeds = self.context_embedder(encoder_hidden_states.to(get_parameter_dtype(self.context_embedder)))
        text_embeds = self.token_refiner(text_embeds)
        hidden_states = text_embeds.new_zeros((text_embeds.shape[0], sequence_length, text_embeds.shape[-1]))
        hidden_states = hidden_states.index_copy(1, text_indices, text_embeds)
        hidden_states = hidden_states.index_copy(1, video_indices, video_embeds.to(text_embeds.dtype))
        hidden_states = hidden_states.index_copy(1, audio_indices, audio_embeds.to(text_embeds.dtype))
        temb = self.time_embedding(timestep, endpoint_timestep)
        adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, temb, adaln_indices, rotary_emb)
        hidden_states = self.norm_out(hidden_states, temb, timestep_indices).to(get_parameter_dtype(self.proj_out))
        video_output = self.proj_out(hidden_states).index_select(1, video_indices)
        audio_output = self.audio_proj_out(hidden_states).index_select(1, audio_indices)
        if not return_dict:
            return (video_output, audio_output)
        return MiniMaxH3TransformerOutput(sample=video_output, audio_sample=audio_output)


def _safetensors_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    for name in ("model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json"):
        index = path / name
        if index.exists():
            return [path / f for f in sorted(set(json.loads(index.read_text())["weight_map"].values()))]
    return sorted(path.glob("*.safetensors"))


def config_from_dict(model_config: dict[str, Any], endpoint_mode: str) -> dict[str, Any]:
    """A checkpoint `config.json` (or student metadata `model_config`) -> `H3TinyTransformer3DModel` kwargs."""
    keys = (
        "num_attention_heads", "attention_head_dim", "hidden_size", "num_layers", "num_refiner_layers", "ffn_dim",
        "refiner_ffn_dim", "in_channels", "audio_in_channels", "patch_size", "text_dim", "freq_dim",
        "time_embed_hidden_dim", "time_embed_dim", "rope_freq_dim", "rope_theta", "norm_eps", "qk_norm_eps",
        "final_norm_eps", "adaln_rank",
    )
    refiner_heads = model_config.get("refiner_num_attention_heads")
    if refiner_heads not in (None, model_config["num_attention_heads"]):
        raise ValueError(f"refiner_num_attention_heads {refiner_heads} != num_attention_heads is not supported")
    config = {k: model_config[k] for k in keys if k in model_config}
    config["patch_size"] = tuple(config.get("patch_size", (1, 2, 2)))
    config["endpoint_mode"] = endpoint_mode
    return config


def build_h3tiny(weights: str | Path, *, endpoint_mode: str) -> tuple[H3TinyTransformer3DModel, list[Path]]:
    """The model on the meta device for a checkpoint dir (`config.json` + safetensors, or `metadata.json` +
    payload), and the safetensors files holding its weights."""
    weights = Path(weights)
    if (weights / "metadata.json").exists():
        meta = json.loads((weights / "metadata.json").read_text())
        config = config_from_dict(meta["model_config"], endpoint_mode)
        files = [weights / meta["payload_file"]]
    else:
        root = next((weights / sub for sub in ("transformer", "transformer_ref") if (weights / sub).is_dir()), weights)
        raw = json.loads((root / "config.json").read_text())
        config = config_from_dict(raw, endpoint_mode)
        files = _safetensors_files(root)
    with torch.device("meta"):
        model = H3TinyTransformer3DModel(**config)
    return model, files


def read_state(files: list[Path], device: str | torch.device = "cpu") -> dict[str, Tensor]:
    state: dict[str, Tensor] = {}
    for file in files:
        with safe_open(str(file), framework="pt", device=str(device)) as f:
            for key in f.keys():
                state[key] = f.get_tensor(key)
    return state


def stored_dtypes(files: list[Path]) -> dict[str, torch.dtype]:
    """Stored dtype of every tensor, read from the safetensors headers only."""
    names = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64}
    out = {}
    for file in files:
        with open(file, "rb") as f:
            header = json.loads(f.read(int.from_bytes(f.read(8), "little")))
        out.update({k: names[v["dtype"]] for k, v in header.items() if k != "__metadata__"})
    return out


def check_state(model: nn.Module, state: dict[str, Tensor], weights: str | Path) -> None:
    expected = dict(model.state_dict())
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    shapes = [(k, tuple(state[k].shape), tuple(expected[k].shape)) for k in set(state) & set(expected)
              if tuple(state[k].shape) != tuple(expected[k].shape)]
    if missing or unexpected or shapes:
        raise RuntimeError(f"weights {weights} do not match: missing={missing[:8]} ({len(missing)}), "
                           f"unexpected={unexpected[:8]} ({len(unexpected)}), shapes={shapes[:5]}")


def rebuild_rope(model: H3TinyTransformer3DModel, device: str | torch.device) -> None:
    """The RoPE inverse frequencies are a non-persistent buffer built on meta; rebuild it as diffusers does."""
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed

    rope = MiniMaxH3RotaryPosEmbed(model.config.rope_freq_dim, model.config.rope_theta)
    model.rope.register_buffer("inv_freq", rope.inv_freq.to(device), persistent=False)


def load_h3tiny(weights: str | Path, *, endpoint_mode: str, device: str | torch.device = "cpu") -> H3TinyTransformer3DModel:
    """Build from a checkpoint dir, then load every tensor strictly at its stored dtype (bf16 blocks, fp32
    embedders/heads as trained)."""
    model, files = build_h3tiny(weights, endpoint_mode=endpoint_mode)
    state = read_state(files, device)
    check_state(model, state, weights)
    model.load_state_dict(state, assign=True, strict=True)
    rebuild_rope(model, device)
    leftover = [name for name, buffer in model.named_buffers() if buffer.is_meta]
    if leftover:
        raise RuntimeError(f"meta buffers left after load: {leftover}")
    return model.eval().requires_grad_(False)


def save_h3tiny(state: dict[str, Tensor], out: str | Path, config: dict[str, Any], shard_bytes: int = 8 << 30,
                extra: dict[str, Any] | None = None) -> Path:
    """Write ``state`` as sharded safetensors with ``model.safetensors.index.json`` and ``config.json`` (the
    layout ``load_h3tiny`` reads), tensors at their given dtypes. ``extra`` goes to ``provenance.json``."""
    from safetensors.torch import save_file

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    shards, current, size = [], {}, 0
    for name, tensor in state.items():
        nbytes = tensor.numel() * tensor.element_size()
        if current and size + nbytes > shard_bytes:
            shards.append(current)
            current, size = {}, 0
        current[name] = tensor.detach().cpu().contiguous()
        size += nbytes
    shards.append(current)
    weight_map = {}
    for i, part in enumerate(shards):
        file = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(part, str(out / file), metadata={"format": "pt"})
        weight_map.update({k: file for k in part})
    total = sum(t.numel() * t.element_size() for t in state.values())
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=1))
    (out / "config.json").write_text(json.dumps(config, indent=1))
    if extra is not None:
        (out / "provenance.json").write_text(json.dumps(extra, indent=1))
    return out


def checkpoint_config(weights: str | Path) -> dict[str, Any]:
    """The raw ``config.json`` of a checkpoint dir (or its ``transformer`` / ``transformer_ref`` subdir)."""
    weights = Path(weights)
    root = next((weights / sub for sub in ("transformer", "transformer_ref") if (weights / sub).is_dir()), weights)
    return json.loads((root / "config.json").read_text())
