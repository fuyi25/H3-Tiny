"""Calibration probe on prepared samples (see `h3tiny.data`): block influence, drop comparisons and module input rows.

The transformer runs on ground-truth latents plus noise at points of the official 32-step shifted schedule, and the
packed sequence is read by row group:

* ``v``: generated video rows, ``a``: generated audio rows,
* ``ref``: reference (conditioning) video rows, ``text``: text rows.

At every (sample, point) a fixed random subset of each group is drawn (128 video / audio rows, 32 reference / text
rows, seeded by ``sample_id:step``) and reused by all measurements of that forward:

* block influence ``BI = mean(1 - cos(block input, block output))`` per block and group (depth pruning),
* the inputs of every block's FFN down projection and attention output projection, video and audio rows (width
  pruning),
* for drop candidates, the relative MSE of the video / audio velocity with those blocks skipped (depth selection).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence

import torch

from .data import forward_velocity, noised_inputs
from .prune.adaln import AUDIO_SHIFT, VIDEO_SHIFT
from .prune.blocks import block_influence

#: Steps of the 32-step schedule used for calibration, and the tuning subset (as for the released 17B / 14B).
CALIBRATION_STEPS = (0, 4, 8, 12, 16, 20, 24, 31)
TUNING_STEPS = (4, 16, 31)
ROWS = {"v": 128, "a": 128, "ref": 32, "text": 32}


def schedule_points(steps: Sequence[int] = CALIBRATION_STEPS, num_steps: int = 32) -> list[tuple[int, float, float]]:
    """``(step, t_video, t_audio)`` at the given steps of the official ``num_steps`` shifted schedule."""
    out = []
    for i in steps:
        sigma = 1 - i / num_steps
        out.append((i, 1 - VIDEO_SHIFT * sigma / (1 + (VIDEO_SHIFT - 1) * sigma),
                    1 - AUDIO_SHIFT * sigma / (1 + (AUDIO_SHIFT - 1) * sigma)))
    return out


def row_groups(prepared: dict) -> dict[str, torch.Tensor]:
    ncv, nca = prepared["num_condition_video_rows"], prepared["num_condition_audio_rows"]
    vi, ai = prepared["video_indices"], prepared["audio_indices"]
    return {"v": vi[ncv:], "a": ai[nca:], "ref": vi[:ncv], "text": prepared["text_indices"]}


def pick_rows(prepared: dict, step: int, device, rows: dict[str, int] = ROWS) -> dict[str, torch.Tensor]:
    seed = int(hashlib.sha256(f"{prepared['sample_id']}:{step}".encode()).hexdigest()[:15], 16)
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for name, ix in row_groups(prepared).items():
        if ix.numel():
            out[name] = ix[torch.randperm(ix.numel(), generator=gen)[:rows[name]]].to(device)
    return out


def _relmse(pred: torch.Tensor, ref: torch.Tensor) -> float:
    return float((pred.float() - ref.float()).square().sum() / ref.float().square().sum().clamp_min(1e-20))


@torch.no_grad()
def probe(transformer, samples: Iterable[dict], points: Sequence[tuple[int, float, float]], *,
          collect_rows: bool = False, row_kinds: Sequence[str] = ("ff", "at"), candidates: Sequence[dict] = (), seed: int = 1234,
          device: torch.device | str = "cuda") -> dict:
    """Run the probe over ``samples`` (an iterable of prepared dicts, e.g. loaded lazily from disk).

    Returns ``bi[i][group] = [bi, ...]`` (one value per sample and point), ``rows[i] = {"ff_v", "ff_a", "at_v",
    "at_a"}`` (bf16, CPU; the kinds in ``row_kinds``) when ``collect_rows``, and ``errors[name] = [{video_relmse, audio_relmse}, ...]`` per
    candidate in ``candidates`` (each ``{"name", "drop_layers"}``).
    """
    blocks = list(transformer.transformer_blocks)
    bi = [{} for _ in blocks]
    rows = [{} for _ in blocks]
    errors = {c["name"]: [] for c in candidates}
    state: dict = {"collect": False, "drop": set()}
    handles = []

    def before(i):
        def hook(module, args):
            if state["collect"]:
                x = args[0][0]
                state[i] = {m: x.index_select(0, ix).float() for m, ix in state["rows"].items()}
        return hook

    def after(i):
        def hook(module, args, output):
            if i in state["drop"]:
                return args[0]
            if state["collect"]:
                z = output[0]
                for m, x in state.pop(i).items():
                    bi[i].setdefault(m, []).append(block_influence(x, z.index_select(0, state["rows"][m])))
        return hook

    def capture(i, kind):
        def hook(module, args):
            if state["collect"] and collect_rows:
                x = args[0][0]
                for m in ("v", "a"):
                    rows[i].setdefault(f"{kind}_{m}", []).append(
                        x.index_select(0, state["rows"][m]).to("cpu", torch.bfloat16))
        return hook

    for i, block in enumerate(blocks):
        handles += [block.register_forward_pre_hook(before(i)), block.register_forward_hook(after(i))]
        if collect_rows and "ff" in row_kinds:
            handles.append(block.ff.net[-1].register_forward_pre_hook(capture(i, "ff")))
        if collect_rows and "at" in row_kinds:
            handles.append(block.attn.to_out[0].register_forward_pre_hook(capture(i, "at")))
    try:
        for prepared in samples:
            gen = torch.Generator().manual_seed(seed)
            for step, tv, ta in points:
                kwargs, _ = noised_inputs(prepared, tv, ta, gen, device)
                state.update(collect=True, drop=set(), rows=pick_rows(prepared, step, device))
                base_v, base_a = forward_velocity(transformer, kwargs, prepared)
                state["collect"] = False
                for c in candidates:
                    state["drop"] = set(c["drop_layers"])
                    v, a = forward_velocity(transformer, kwargs, prepared)
                    errors[c["name"]].append(dict(sample_id=prepared["sample_id"], step=step,
                                                  video_relmse=_relmse(v, base_v), audio_relmse=_relmse(a, base_a)))
                state["drop"] = set()
    finally:
        for h in handles:
            h.remove()
    return dict(bi=bi, rows=[{k: torch.cat(v) for k, v in r.items()} for r in rows] if collect_rows else None,
                errors=errors)
