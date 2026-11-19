"""End-to-end smoke test on CPU with a tiny random "official" model: prune_depth -> prune_width.

    python tests/smoke_pipeline.py /tmp/h3tiny-smoke    # needs diffusers with MiniMax-H3; no GPU, no real weights

Checks that the depth stage drops the planned number of blocks and factors AdaLN, the width stage narrows the FFNs,
both outputs load with `load_h3tiny`, and a full-rank depth export reproduces the teacher with the same blocks skipped.
"""
import json
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from h3tiny.data import forward_velocity, noised_inputs  # noqa: E402
from h3tiny.modeling import H3TinyTransformer3DModel, load_h3tiny  # noqa: E402

FP32 = ("proj_in", "audio_proj_in", "proj_out", "audio_proj_out", "time_embedder", "r_time_embedder")
BASE = dict(num_attention_heads=2, attention_head_dim=128, hidden_size=256, num_refiner_layers=1, text_dim=64,
            freq_dim=256, time_embed_hidden_dim=256, time_embed_dim=128)


def save(model: H3TinyTransformer3DModel, out: Path, extra: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    state = {k: v.to(torch.float32 if k.split(".")[0] in FP32 else torch.bfloat16).contiguous()
             for k, v in model.state_dict().items()}
    save_file(state, str(out / "model.safetensors"))
    (out / "config.json").write_text(json.dumps({**BASE, **extra}))


def make_samples(directory: Path, n: int, proj_in: int, audio_in: int, seed: int) -> None:
    from diffusers.models.transformers.transformer_minimax_h3 import MINIMAX_H3_MODALITY_NUM

    directory.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    for i in range(n):
        n_text, n_audio, ncv, n_gen = 8, 40, 6, 64 + 4 * i
        seq = n_text + n_audio + ncv + n_gen
        prepared = dict(
            latents=torch.randn(ncv + n_gen, proj_in, generator=g),
            audio_latents=torch.randn(n_audio, audio_in, generator=g),
            prompt_embeds=torch.randn(1, n_text, 64, generator=g).bfloat16(),
            num_condition_video_rows=ncv, num_condition_audio_rows=0, keyframe_noise_aug=0.999,
            token_tags=torch.arange(seq) % MINIMAX_H3_MODALITY_NUM,
            position_ids=torch.stack([torch.arange(seq, dtype=torch.float64)] * 3, -1),
            text_indices=torch.arange(n_text), audio_indices=torch.arange(n_text, n_text + n_audio),
            video_indices=torch.arange(n_text + n_audio, seq), sample_id=f"{directory.name}{i}")
        torch.save(prepared, directory / f"{prepared['sample_id']}.pt")


def run(*cmd, env=None) -> None:
    subprocess.run([sys.executable, *map(str, cmd)], check=True, env=env)


def main(work: Path) -> None:
    torch.manual_seed(0)
    teacher = H3TinyTransformer3DModel(**BASE, num_layers=8, ffn_dim=512, endpoint_mode="none")
    save(teacher, work / "teacher", dict(num_layers=8, ffn_dim=512))
    make_samples(work / "calibration", 3, teacher.proj_in.in_features, teacher.audio_proj_in.in_features, 1)
    make_samples(work / "tuning", 2, teacher.proj_in.in_features, teacher.audio_proj_in.in_features, 2)

    run(ROOT / "scripts/prune_depth.py", "--base", work / "teacher", "--calibration", work / "calibration",
        "--tuning", work / "tuning", "--out", work / "depth", "--num-drop", 2, "--protect", 2, "--adaln-rank", 4,
        "--device", "cpu")
    depth = load_h3tiny(work / "depth", endpoint_mode="none")
    plan = json.loads((work / "depth/plan/depth_plan.json").read_text())
    assert depth.config.num_layers == 6 and depth.config.adaln_rank == 4 and len(plan["drop_layers"]) == 2, plan
    run(ROOT / "scripts/prune_width.py", "--parent", work / "depth", "--calibration", work / "calibration",
        "--tuning", work / "tuning", "--out", work / "width", "--ffn-dim", 384, "--device", "cpu")
    width = load_h3tiny(work / "width", endpoint_mode="none")
    assert width.config.ffn_dim == 384 and width.transformer_blocks[0].ff.net[2].in_features == 384
    assert width.token_refiner is not None
    print("pruning:", plan["drop_layers"], "dropped;", sum(p.numel() for p in teacher.parameters()), "->",
          sum(p.numel() for p in depth.parameters()), "->", sum(p.numel() for p in width.parameters()), "params")
    # Full-rank AdaLN with the same plan must reproduce the teacher with those blocks skipped (up to bf16 storage).
    run(ROOT / "scripts/prune_depth.py", "--base", work / "teacher", "--out", work / "depth_full", "--drop",
        *plan["drop_layers"], "--adaln-rank", BASE["time_embed_dim"], "--device", "cpu")
    full = load_h3tiny(work / "depth_full", endpoint_mode="none")
    ref_model = load_h3tiny(work / "teacher", endpoint_mode="none")
    for i in plan["drop_layers"]:
        ref_model.transformer_blocks[i].register_forward_hook(lambda m, args, out: args[0])
    prepared = torch.load(next((work / "tuning").glob("*.pt")), weights_only=True)
    kwargs, _ = noised_inputs(prepared, 0.4, 0.6, torch.Generator().manual_seed(0), "cpu")
    with torch.no_grad():
        (v0, a0), (v1, a1) = forward_velocity(ref_model, kwargs, prepared), forward_velocity(full, kwargs, prepared)
    rel = [float((x - y).square().sum() / y.square().sum()) for x, y in ((v1, v0), (a1, a0))]
    print(f"full-rank depth export vs teacher with blocks skipped: video/audio relMSE {rel[0]:.2e} / {rel[1]:.2e}")
    assert max(rel) < 1e-3, rel
    print("pruning smoke test passed")

if __name__ == "__main__":
    main(Path(sys.argv[1]))
