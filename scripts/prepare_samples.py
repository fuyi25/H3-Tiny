"""Encode samples into cached transformer inputs with ground-truth targets (for calibration and OKD).

Input is a JSONL manifest, one sample per line:

    {"sample_id": "0001", "prompt": "...", "references": ["ref1.png", "ref2.png"], "video": "gt.mp4",
     "num_frames": 124}

``"workflow"`` is ``"ref2va"`` by default (for the Ref2VA transformer). For the FL2VA transformer use ``"fl2va"``
(keyframes are the first and last frame of ``video``; no references) and ``"t2va"`` (text only); a manifest may mix
the two.

Writes ``<out>/<sample_id>.pt``; existing files are skipped, so the job can be resumed. Shard over GPUs with
``torchrun --nproc_per_node 8 scripts/prepare_samples.py ...`` (each rank takes every world_size-th sample), or run
processes with ``--shard i --num-shards n``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h3tiny.data import H3Batcher, Sample  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", required=True, help="official MiniMax-H3 diffusers directory")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--shard", type=int, default=0, help="without torchrun: this process's shard")
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    # Trust RANK / WORLD_SIZE only under torchrun: cluster schedulers often export them for their own jobs.
    if "TORCHELASTIC_RUN_ID" in os.environ:
        rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world, local = args.shard, args.num_shards, 0
    torch.cuda.set_device(local)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    samples = [Sample(**json.loads(line)) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    todo = [s for s in samples[rank::world] if not (out / f"{s.sample_id}.pt").exists()]
    print(f"[rank {rank}] {len(todo)} of {len(samples[rank::world])} samples to encode", flush=True)
    if not todo:
        return
    todo.sort(key=lambda s: s.workflow)  # one batcher per workflow
    batcher = None
    failed = 0
    for i, sample in enumerate(todo):
        if batcher is None or batcher.workflow != sample.workflow:
            batcher = None
            torch.cuda.empty_cache()
            batcher = H3Batcher(args.base_model, device=f"cuda:{local}", workflow=sample.workflow)
        try:
            prepared = batcher.prepare(sample, seed=args.seed)
        except Exception:  # keep going; report at the end
            failed += 1
            print(f"[rank {rank}] FAILED {sample.sample_id}\n{traceback.format_exc()}", flush=True)
            continue
        tmp = out / f".{sample.sample_id}.pt.rank{rank}"
        torch.save(prepared, tmp)
        tmp.rename(out / f"{sample.sample_id}.pt")
        print(f"[rank {rank}] {i + 1}/{len(todo)} {sample.sample_id}", flush=True)
    if failed:
        raise SystemExit(f"[rank {rank}] {failed} samples failed")


if __name__ == "__main__":
    main()
