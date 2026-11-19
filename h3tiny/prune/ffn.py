"""Stage 3: FFN width pruning with audio-aware, parent-anchored ridge compensation.

For each block we keep ``width`` of the FFN's hidden channels and refit the down projection on them so the block
output stays close to the parent's. Inputs are activation rows collected on calibration data, split by modality:
video and audio tokens are scored and fitted separately, so the (much smaller) audio stream is not drowned out.

* Channel score: rms activation x column norm of the down projection, normalized per modality, audio weighted.
* Refit: ridge regression of the kept columns, anchored at the parent weights (only the correction is solved).
* Strength: the correction is scaled by alpha in {0, .5, 1}, chosen on a separate tuning split.
"""

from __future__ import annotations

import torch


def choose(xv: torch.Tensor, xa: torch.Tensor, weight: torch.Tensor, width: int, head_dim: int = 1,
           audio_weight: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Pick ``width`` input channels (or whole heads when ``head_dim > 1``) of ``weight`` (out x in).

    Returns the kept input columns, the kept channel/head indices, and the importance mass they retain.
    """
    rv, ra = xv.float().square().mean(0), xa.float().square().mean(0)
    if head_dim > 1:
        rv, ra = rv.reshape(-1, head_dim).mean(1), ra.reshape(-1, head_dim).mean(1)
        norms = weight.float().reshape(weight.shape[0], -1, head_dim).square().sum((0, 2)).sqrt()
    else:
        norms = weight.float().norm(dim=0)
    v, a = rv.sqrt() * norms, ra.sqrt() * norms
    if not (v.sum() > 0 and a.sum() > 0 and torch.isfinite(v).all() and torch.isfinite(a).all()):
        raise ValueError("channel importances must be finite and non-zero for both modalities")
    indices = (v / v.sum() + audio_weight * a / a.sum()).topk(width).indices.sort().values
    cols = (indices[:, None] * head_dim + torch.arange(head_dim)[None, :]).reshape(-1)
    mass = dict(video_importance_mass=float(v[indices].sum() / v.sum()),
                audio_importance_mass=float(a[indices].sum() / a.sum()))
    return cols, indices, mass


def energy(x: torch.Tensor, weight: torch.Tensor, device: torch.device | str) -> float:
    """Mean squared output of ``x @ weight.T`` per element (the normalizer of a modality's objective)."""
    w = weight.to(device=device, dtype=torch.float32)
    total = 0.0
    for c in x.split(1024):
        total += float((c.to(device=device, dtype=torch.float32) @ w.T).square().sum())
    return max(total / (len(x) * weight.shape[0]), 1e-20)


def gram(xv: torch.Tensor, xa: torch.Tensor, weight: torch.Tensor, device: torch.device | str = "cpu",
         relative: bool = True, audio_weight: float = 1.0) -> tuple[torch.Tensor, dict[str, float]]:
    """Input Gram of the refit objective; each modality is normalized by its own output energy when ``relative``."""
    n = xv.shape[1]
    g = torch.zeros(n, n, device=device, dtype=torch.float32)
    ev = energy(xv, weight, device) if relative else 1.0
    ea = energy(xa, weight, device) if relative else 1.0
    for x, mass, e in [(xv, 1.0, ev), (xa, audio_weight, ea)]:
        scale = mass / ((1 + audio_weight) * len(x) * e)
        for c in x.split(2048):
            z = c.to(device=device, dtype=torch.float32)
            g.addmm_(z.T, z, alpha=scale)
    return g, dict(video_output_energy=ev, audio_output_energy=ea, relative_objective=relative, audio_weight=audio_weight)


def fit(weight: torch.Tensor, g: torch.Tensor, cols: torch.Tensor, ridge: float = 1e-3,
        device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Ridge refit of the kept columns, anchored at the parent: returns ``(prior, delta, info)``.

    ``prior`` is the parent restricted to ``cols``; ``prior + delta`` minimizes the Gram-weighted output error plus
    ``ridge * mean(diag)`` times the squared distance from ``prior``.
    """
    full = weight.to(device=device, dtype=torch.float64)
    g = g.to(device=device, dtype=torch.float64)
    k = cols.to(device)
    q = g[k][:, k]
    prior = full[:, k]
    rhs = full @ g[:, k] - prior @ q
    reg = ridge * torch.diagonal(q).mean().clamp_min(1e-12)
    q.diagonal().add_(reg)
    chol, info = torch.linalg.cholesky_ex(q)
    if int(info):
        raise RuntimeError(f"Cholesky failed with info={int(info)}")
    delta = torch.cholesky_solve(rhs.T.contiguous(), chol).T
    residual = float((delta @ q - rhs).norm() / rhs.norm().clamp_min(1e-20))
    if not (residual < 1e-7 and torch.isfinite(delta).all()):
        raise RuntimeError(f"ridge solve is inaccurate (relative residual {residual})")
    return prior.cpu(), delta.cpu(), dict(solver_relative_residual=residual, ridge_scaled=float(reg))


def errors(xv: torch.Tensor, xa: torch.Tensor, full: torch.Tensor, cols: torch.Tensor, narrow: torch.Tensor,
           device: torch.device | str = "cpu") -> dict[str, float]:
    """Relative output MSE of the narrowed projection against the parent, per modality (and worst chunk)."""
    f, w, k = full.to(device), narrow.to(device), cols.to(device)
    report = {}
    for name, x in [("video", xv), ("audio", xa)]:
        numerator = denominator = 0.0
        chunks = []
        for c in x.split(512):
            z = c.to(device=device, dtype=f.dtype)
            y = torch.nn.functional.linear(z, f).float()
            p = torch.nn.functional.linear(z[:, k], w).float()
            n, d = float((p - y).square().sum()), float(y.square().sum())
            numerator += n
            denominator += d
            chunks.append(n / max(d, 1e-20))
        report[name + "_relmse"] = numerator / max(denominator, 1e-20)
        report[name + "_worst_chunk_relmse"] = max(chunks)
    return report


def choose_strength(xv: torch.Tensor, xa: torch.Tensor, full: torch.Tensor, cols: torch.Tensor, prior: torch.Tensor,
                    delta: torch.Tensor, device: torch.device | str = "cpu") -> tuple[torch.Tensor, dict]:
    """Pick alpha in {0, .5, 1}: the lowest tuning audio error among alphas within the video error budget."""
    candidates = []
    for alpha in (0.0, 0.5, 1.0):
        w = (prior + alpha * delta).to(full.dtype)
        candidates.append(dict(alpha=alpha, **errors(xv, xa, full, cols, w, device)))
    best_video = min(x["video_relmse"] for x in candidates)
    eligible = [x for x in candidates if x["video_relmse"] <= max(best_video * 1.25, best_video + 0.001)]
    selected = min(eligible, key=lambda x: (x["audio_relmse"], x["audio_worst_chunk_relmse"]))
    w = (prior + selected["alpha"] * delta).to(full.dtype)
    return w, dict(selected=selected, candidates=candidates, selection="minimum audio tuning relMSE within video budget")


def narrow_ffn(up: torch.Tensor, down_new: torch.Tensor, kept: torch.Tensor, ffn_dim: int) -> dict[str, torch.Tensor]:
    """Slice the SwiGLU up projection (both halves, same channels) and pair it with the refit down projection."""
    return {"ff.net.0.proj.weight": torch.cat([up[:ffn_dim][kept], up[ffn_dim:][kept]]), "ff.net.2.weight": down_new}
