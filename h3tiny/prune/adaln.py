"""Stage 1: data-free low-rank AdaLN.

Every MiniMax-H3 block modulates its rows with ``adaln_proj.linear(silu(temb))``: a 2688 -> 96768 projection
(about 260M parameters per block, 13B over 50 blocks). Its input depends on the timestep alone, so it lives on a
one-dimensional curve and the projection can be factored through a tiny bottleneck with almost no output error.

We sample that input on a timestep grid, whiten it, and keep the top-``rank`` singular directions of the
whitened projection: ``W ~= second @ first`` with ``first`` (rank x 2688) and ``second`` (96768 x rank). This
minimizes the output error on the sampled inputs, not the weight error. No data, no training.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

#: Official H3 flow shifts for video and audio.
VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0
#: Reference rows are held at this timestep.
REFERENCE_TIMESTEP = 0.999


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding with ``flip_sin_to_cos=True, downscale_freq_shift=0`` (diffusers / H3 ``Timesteps``)."""
    half = dim // 2
    exponent = -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    emb = torch.cat([emb[:, half:], emb[:, :half]], dim=-1)
    if dim % 2:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


def schedule_timesteps(num_steps: int = 32, indices: tuple[int, ...] = (0, 4, 8, 12, 16, 20, 24, 31)) -> list[float]:
    """Video and audio timesteps (t = 1 - shifted sigma) at a subsample of the official ``num_steps`` schedule."""
    out = []
    for i in indices:
        sigma = 1 - i / num_steps
        out.append(1 - VIDEO_SHIFT * sigma / (1 + (VIDEO_SHIFT - 1) * sigma))
        out.append(1 - AUDIO_SHIFT * sigma / (1 + (AUDIO_SHIFT - 1) * sigma))
    return out


def default_timesteps(device: torch.device | str = "cpu") -> torch.Tensor:
    """512 uniform timesteps plus the exact schedule points and the reference timestep, deduplicated."""
    grid = torch.linspace(0.0005, 0.9995, 512, device=device)
    extra = torch.tensor(schedule_timesteps() + [REFERENCE_TIMESTEP], device=device)
    return torch.unique(torch.cat([grid, extra]))


def adaln_inputs(time_embedder: Mapping[str, torch.Tensor], freq_dim: int, timesteps: torch.Tensor) -> torch.Tensor:
    """The AdaLN input ``silu(time_embedder(proj(t)))`` for every timestep, in float64.

    ``time_embedder`` holds ``linear_1.weight/bias`` and ``linear_2.weight/bias`` of the model's ``time_embedder``.
    """
    dev = timesteps.device
    w1, b1 = (time_embedder[f"linear_1.{k}"].float().to(dev) for k in ("weight", "bias"))
    w2, b2 = (time_embedder[f"linear_2.{k}"].float().to(dev) for k in ("weight", "bias"))
    e = timestep_embedding(timesteps, freq_dim).float()
    h = torch.nn.functional.silu(torch.addmm(b1, e, w1.T))
    return torch.nn.functional.silu(torch.addmm(b2, h, w2.T)).double()


class AdaLNFactorizer:
    """Whitens the sampled AdaLN inputs once, then factors any block's ``adaln_proj.linear``."""

    def __init__(self, inputs: torch.Tensor, rel_tol: float = 1e-5):
        self.inputs = inputs.double()
        # Compact whitening, algebraically W C^(1/2) without the zero-covariance directions.
        _, s, vh = torch.linalg.svd(self.inputs / math.sqrt(len(self.inputs)), full_matrices=False)
        valid = s > s.max() * rel_tol
        s, vh = s[valid], vh[valid]
        self.basis = vh.T * s
        self.inverse_basis = vh / s[:, None]
        self.covariance_rank = int(len(s))

    def factorize(self, weight: torch.Tensor, bias: torch.Tensor, rank: int, dtype: torch.dtype = torch.bfloat16
                  ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        """Return ``{linear_1.weight, linear_1.bias, linear_2.weight, linear_2.bias}`` and output-error metrics.

        ``linear_1`` has a zero bias; ``linear_2`` keeps the original bias, so ``t``-independent offsets are exact.
        """
        dev = self.basis.device
        w = weight.double().to(dev)
        u, sv, v = torch.linalg.svd(w @ self.basis, full_matrices=False)
        first = v[:rank] @ self.inverse_basis
        second = u[:, :rank] * sv[:rank]
        target = self.inputs @ w.T
        pred = (self.inputs @ first.T) @ second.T
        fb, sb = first.to(dtype), second.to(dtype)
        cast = torch.nn.functional.linear(torch.nn.functional.linear(self.inputs.to(dtype), fb), sb).double()
        metrics = dict(fp64_rel_l2=float((target - pred).norm() / target.norm()),
                       cast_rel_l2=float((cast - target).norm() / target.norm()),
                       covariance_rank=self.covariance_rank)
        tensors = {"linear_1.weight": fb.cpu(), "linear_1.bias": torch.zeros(rank, dtype=dtype),
                   "linear_2.weight": sb.cpu(), "linear_2.bias": bias.detach().cpu().to(dtype)}
        return tensors, metrics
