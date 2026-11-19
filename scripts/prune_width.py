"""Stage 3: 17B -> 14B. Narrow every block's FFN (14336 -> 9792 channels) with audio-aware ridge compensation.

    python scripts/prune_width.py --parent /models/h3tiny-17b-pruned \\
        --calibration /cache/calibration --tuning /cache/tuning --out /models/h3tiny-14b-pruned

Run on the *pruned 17B* (output of ``scripts/prune_depth.py``), so the rows reflect the depth-pruned network. Per block:

1. Collect inputs of the FFN down projection ``ff.net.2`` on generated video and audio rows (128 rows of each per
   sample and schedule point; calibration: 8 points, tuning: 3 points).
2. Keep the ``--ffn-dim`` channels with the largest modality-balanced output energy (``ffn.choose``).
3. Refit the down projection on the kept channels: ridge regression toward the parent's outputs, video and audio each
   normalized by their own output energy and audio weighted 2x, anchored at the parent's own columns (``ffn.fit``).
4. On the tuning rows, pick the compensation strength alpha in {0, 0.5, 1} with the lowest audio error within the
   video error budget (``ffn.choose_strength``); slice the SwiGLU up projection to the same channels.

Attention heads are kept (our width screen preferred FFN-only narrowing at equal parameter budget). No training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h3tiny.calibrate import CALIBRATION_STEPS, TUNING_STEPS, probe, schedule_points  # noqa: E402
from h3tiny.modeling import build_h3tiny, checkpoint_config, load_h3tiny, read_state, save_h3tiny  # noqa: E402
from h3tiny.prune import ffn  # noqa: E402

from prune_depth import samples  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parent", required=True, help="depth-pruned checkpoint (scripts/prune_depth.py output)")
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--tuning", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ffn-dim", type=int, default=9792)
    ap.add_argument("--audio-weight", type=float, default=2.0, help="audio weight of the refit objective")
    ap.add_argument("--selection-audio", type=float, default=1.0, help="audio weight of the channel ranking")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False

    config = checkpoint_config(args.parent)
    if config.get("refiner_ffn_dim") is None:
        config["refiner_ffn_dim"] = config["ffn_dim"]
    model = load_h3tiny(args.parent, endpoint_mode="none", device=args.device)
    train = probe(model, samples(args.calibration), schedule_points(CALIBRATION_STEPS), collect_rows=True,
                  row_kinds=("ff",), device=args.device)["rows"]
    tune = probe(model, samples(args.tuning), schedule_points(TUNING_STEPS), collect_rows=True,
                  row_kinds=("ff",), device=args.device)["rows"]
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _, files = build_h3tiny(args.parent, endpoint_mode="none")
    state = read_state(files)
    d, metrics, selection = config["ffn_dim"], [], {}
    for i in range(config["num_layers"]):
        prefix = f"transformer_blocks.{i}."
        down, up = state[prefix + "ff.net.2.weight"], state[prefix + "ff.net.0.proj.weight"]
        xv, xa = train[i]["ff_v"].float(), train[i]["ff_a"].float()
        tv, ta = tune[i]["ff_v"].float(), tune[i]["ff_a"].float()
        cols, _, mass = ffn.choose(xv, xa, down, args.ffn_dim, 1, args.selection_audio)
        g, objective = ffn.gram(xv, xa, down, device=args.device, audio_weight=args.audio_weight)
        prior, delta, solver = ffn.fit(down, g, cols, device=args.device)
        new_down, choice = ffn.choose_strength(tv, ta, down, cols, prior, delta, device=args.device)
        state.update({prefix + k: v.to(down.dtype) for k, v in ffn.narrow_ffn(up, new_down, cols, d).items()})
        selection[str(i)] = cols.tolist()
        metrics.append(dict(layer=i, mass=mass, solver=solver, objective=objective, **choice))
        s = choice["selected"]
        print(f"block {i:2d}: alpha={s['alpha']} video relMSE {s['video_relmse']:.4f} "
              f"audio relMSE {s['audio_relmse']:.4f}", flush=True)
        train[i] = tune[i] = None  # free the rows as we go
    config.update(ffn_dim=args.ffn_dim)
    out = Path(args.out)
    params = sum(t.numel() for t in state.values())
    save_h3tiny(state, out, config, extra=dict(stage="ffn-width", parent=str(args.parent), metrics=metrics,
                                               parameters=params))
    (out / "selection.json").write_text(json.dumps(selection))
    print(f"wrote {out} ({params / 1e9:.2f}B parameters)", flush=True)


if __name__ == "__main__":
    main()
