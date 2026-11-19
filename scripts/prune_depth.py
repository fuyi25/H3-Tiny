"""Stages 1-2: official 33B -> 17B. Drop 8 of 50 blocks by measured block influence; factor AdaLN to rank 8.

    python scripts/prune_depth.py --base /models/MiniMax-H3 \\
        --calibration /cache/calibration --tuning /cache/tuning --out /models/h3tiny-17b-pruned

``--calibration`` / ``--tuning`` hold ``*.pt`` samples from ``scripts/prepare_samples.py`` (we used 24 / 8 disjoint
clips). One GPU with room for the 33B teacher in bf16 (~66 GB) is enough. Steps, all written to ``--out/plan``:

1. ``bi.json``: block influence per block and row group on the calibration samples at 8 schedule points.
2. ``candidates.json``: protect the 12 blocks with the highest audio BI (p90), then one drop set of 8 blocks per audio
   weight in {1, 2, 4}, ranked by ``w * BI_audio + BI_video + 0.1 (BI_ref + BI_text)`` (each p90, scale-normalized).
3. ``depth_plan.json``: forward every candidate with its blocks skipped on the tuning samples (3 schedule points);
   among candidates within 25% (or +0.002) of the best mean video relMSE, keep the lowest audio p90.
4. Export: the kept blocks renumbered, each ``adaln_proj.linear`` replaced by a rank-8 data-free factorization
   whitened over the time-embedding inputs of the whole schedule (``h3tiny.prune.adaln``). No training.

The released 17B dropped blocks [6, 7, 8, 9, 10, 11, 13, 15]; pass ``--drop`` to reuse a plan and skip 1-3.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h3tiny.calibrate import CALIBRATION_STEPS, TUNING_STEPS, probe, schedule_points  # noqa: E402
from h3tiny.modeling import build_h3tiny, checkpoint_config, load_h3tiny, read_state, save_h3tiny  # noqa: E402
from h3tiny.prune import adaln, blocks  # noqa: E402


def samples(directory: str):
    files = sorted(Path(directory).glob("*.pt"))
    if not files:
        raise SystemExit(f"no *.pt samples in {directory}")
    for f in files:
        yield torch.load(f, map_location="cpu", weights_only=True)


def plan_depth(args, plan_dir: Path, num_layers: int) -> dict:
    model = load_h3tiny(args.base, endpoint_mode="none", device=args.device)
    t0 = time.time()
    result = probe(model, samples(args.calibration), schedule_points(CALIBRATION_STEPS), device=args.device)
    layers = blocks.summarize(result["bi"])
    (plan_dir / "bi.json").write_text(json.dumps(dict(layers=layers, raw=result["bi"])))
    print(f"block influence measured in {time.time() - t0:.0f}s", flush=True)
    cand = blocks.depth_candidates(layers, drop=args.num_drop, protect=args.protect)
    (plan_dir / "candidates.json").write_text(json.dumps(cand, indent=1))
    print("candidates:", [c["drop_layers"] for c in cand["candidates"]], flush=True)
    errors = probe(model, samples(args.tuning), schedule_points(TUNING_STEPS), candidates=cand["candidates"],
                   device=args.device)["errors"]
    plan = blocks.select_depth_plan(cand["candidates"], errors, num_layers)
    (plan_dir / "depth_plan.json").write_text(json.dumps(dict(plan, errors=errors), indent=1))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return plan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="official MiniMax-H3 checkpoint (diffusers dir)")
    ap.add_argument("--calibration")
    ap.add_argument("--tuning")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-drop", type=int, default=8)
    ap.add_argument("--protect", type=int, default=12)
    ap.add_argument("--adaln-rank", type=int, default=8)
    ap.add_argument("--drop", type=int, nargs="*", help="skip measurement and drop these blocks")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False

    out = Path(args.out)
    plan_dir = out / "plan"
    plan_dir.mkdir(parents=True, exist_ok=True)
    config = checkpoint_config(args.base)
    n = config["num_layers"]
    if args.drop is not None:
        plan = dict(drop_layers=sorted(args.drop), kept_layers=[i for i in range(n) if i not in set(args.drop)],
                    source="--drop")
    else:
        if not (args.calibration and args.tuning):
            raise SystemExit("pass --calibration and --tuning samples, or --drop")
        plan = plan_depth(args, plan_dir, n)
    kept = plan["kept_layers"]
    print(f"dropping {plan['drop_layers']}, keeping {len(kept)} of {n} blocks", flush=True)

    _, files = build_h3tiny(args.base, endpoint_mode="none")
    state = blocks.remap_blocks(read_state(files), kept)
    te = {k[len("time_embedder."):]: v.float() for k, v in state.items() if k.startswith("time_embedder.")}
    inputs = adaln.adaln_inputs(te, config["freq_dim"], adaln.default_timesteps()).to(args.device)
    factorizer = adaln.AdaLNFactorizer(inputs)
    rank = min(args.adaln_rank, factorizer.covariance_rank)  # rank above the input covariance rank adds nothing
    metrics = []
    for i in range(len(kept)):
        prefix = f"transformer_blocks.{i}.adaln_proj."
        weight, bias = state.pop(prefix + "linear.weight"), state.pop(prefix + "linear.bias")
        tensors, m = factorizer.factorize(weight, bias, rank, dtype=weight.dtype)
        state.update({prefix + k: v for k, v in tensors.items()})
        metrics.append(dict(new_layer=i, old_layer=kept[i], **m))
        print(f"block {kept[i]:2d} -> {i:2d}: AdaLN rank {rank} rel err {m['fp64_rel_l2']:.2e}", flush=True)
    config.update(num_layers=len(kept), adaln_rank=rank, refiner_ffn_dim=config["ffn_dim"])
    params = sum(t.numel() for t in state.values())
    save_h3tiny(state, out, config, extra=dict(stage="depth+adaln", source=str(args.base), depth_plan=plan,
                                               adaln_metrics=metrics, parameters=params))
    print(f"wrote {out} ({params / 1e9:.2f}B parameters)", flush=True)


if __name__ == "__main__":
    main()
