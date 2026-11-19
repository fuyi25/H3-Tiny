"""Stage 2: depth pruning by per-modality Block Influence, with audio-sensitive blocks protected.

Block Influence (BI) of a block on a set of rows is ``mean(1 - cos(input, output))``: how much the block rotates
the residual stream. It is measured separately on video, audio, reference and text rows of calibration samples
(ground-truth latents plus noise at a few schedule timesteps). A block with low BI everywhere is close to an
identity and can be removed.

Selection, as used for the 17B:

1. Protect the ``protect`` blocks with the highest audio p90 BI; audio is the most fragile modality.
2. For each audio weight ``w`` in ``audio_weights``, score the rest by
   ``w * audio + video + 0.1 * reference + 0.1 * text`` (each p90 normalized by its mean over blocks) and take the
   ``drop`` lowest as a candidate set.
3. Remove each candidate set for real on a separate tuning split and measure the relative MSE of the model's
   video / audio velocity. Among candidates within 25% (or 0.002 absolute) of the best video error, keep the one
   with the lowest audio p90 error.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import mean

import torch

MODALITIES = ("a", "v", "ref", "text")


def block_influence(x: torch.Tensor, z: torch.Tensor) -> float:
    """``mean(1 - cos(x, z))`` over rows; ``x`` and ``z`` are a block's input and output rows (tokens x hidden)."""
    return float((1 - torch.nn.functional.cosine_similarity(x.float(), z.float(), dim=-1)).mean())


def quantile(values: Sequence[float], q: float) -> float:
    """Nearest-rank quantile (no interpolation), as in the original selection."""
    values = sorted(values)
    return values[min(len(values) - 1, math.ceil(q * len(values)) - 1)]


def summarize(measurements: Sequence[Mapping[str, Sequence[float]]]) -> list[dict]:
    """``measurements[i][modality]`` is the list of BI values of block ``i`` over all calibration rows and timesteps.

    Negative BI (a block that sharpens rather than rotates) counts as zero.
    """
    layers = []
    for i, m in enumerate(measurements):
        layer = {"layer": i}
        for mod in MODALITIES:
            values = [max(0.0, x) for x in m[mod]]
            layer[mod] = dict(mean=mean(values), p90=quantile(values, 0.9), maximum=max(values))
        layers.append(layer)
    return layers


def depth_candidates(layers: Sequence[Mapping], drop: int = 8, protect: int = 12,
                     audio_weights: Sequence[float] = (1.0, 2.0, 4.0)) -> dict:
    """Candidate drop sets from the BI summary (see module docstring, steps 1-2)."""
    n = len(layers)
    protected = set(sorted(range(n), key=lambda i: layers[i]["a"]["p90"], reverse=True)[:protect])
    scales = {m: max(mean(x[m]["p90"] for x in layers), 1e-12) for m in MODALITIES}
    candidates, seen = [], set()
    for weight in audio_weights:
        scores = {i: (weight * layers[i]["a"]["p90"] / scales["a"] + layers[i]["v"]["p90"] / scales["v"]
                      + 0.1 * layers[i]["ref"]["p90"] / scales["ref"] + 0.1 * layers[i]["text"]["p90"] / scales["text"])
                  for i in range(n)}
        chosen = tuple(sorted(sorted(set(range(n)) - protected, key=lambda i: scores[i])[:drop]))
        if chosen in seen:
            continue
        seen.add(chosen)
        candidates.append(dict(name=f"audio_weight_{weight:g}", drop_layers=chosen, audio_weight=weight))
    return dict(protected_audio_layers=sorted(protected), candidates=candidates)


def select_depth_plan(candidates: Sequence[Mapping], errors: Mapping[str, Sequence[Mapping[str, float]]],
                      num_layers: int) -> dict:
    """Pick the winner from tuning forward errors (module docstring, step 3).

    ``errors[name]`` lists ``{"video_relmse", "audio_relmse"}`` of candidate ``name`` over tuning cases x timesteps.
    """
    reports = []
    for c in candidates:
        ev = [x["video_relmse"] for x in errors[c["name"]]]
        ea = [x["audio_relmse"] for x in errors[c["name"]]]
        reports.append(dict(c, video_mean=mean(ev), audio_mean=mean(ea),
                            video_p90=quantile(ev, 0.9), audio_p90=quantile(ea, 0.9)))
    best_video = min(x["video_mean"] for x in reports)
    eligible = [x for x in reports if x["video_mean"] <= max(best_video * 1.25, best_video + 0.002)]
    winner = min(eligible, key=lambda x: (x["audio_p90"], x["audio_mean"], x["video_mean"]))
    drop = set(winner["drop_layers"])
    return dict(drop_layers=list(winner["drop_layers"]), kept_layers=[i for i in range(num_layers) if i not in drop],
                winner=winner, candidates=reports)


def remap_blocks(state: Mapping[str, torch.Tensor], kept_layers: Sequence[int]) -> dict[str, torch.Tensor]:
    """Drop the removed blocks and renumber the kept ones contiguously (``transformer_blocks.{new}.*``)."""
    new_index = {old: new for new, old in enumerate(kept_layers)}
    out = {}
    for key, value in state.items():
        if key.startswith("transformer_blocks."):
            old, rest = key[len("transformer_blocks."):].split(".", 1)
            if int(old) not in new_index:
                continue
            key = f"transformer_blocks.{new_index[int(old)]}.{rest}"
        out[key] = value
    return out
