"""CPU unit tests for the three pruning stages (no model weights needed)."""
import torch

from h3tiny.prune import adaln, blocks, ffn


def test_ffn_anchored_ridge_matches_direct_least_squares():
    torch.manual_seed(7)
    xv, xa = torch.randn(80, 9, dtype=torch.float64), 3 * torch.randn(60, 9, dtype=torch.float64)
    w = torch.randn(4, 9, dtype=torch.float64)
    k = torch.tensor([0, 2, 4, 5, 7])
    g = (xv.T @ xv / len(xv) + xa.T @ xa / len(xa)) / 2
    prior, delta, _ = ffn.fit(w, g, k)
    reg = 1e-3 * torch.diagonal(g[k][:, k]).mean()
    design = torch.cat([xv[:, k] / (2 * len(xv)) ** 0.5, xa[:, k] / (2 * len(xa)) ** 0.5,
                        reg.sqrt() * torch.eye(len(k), dtype=torch.float64)])
    target = torch.cat([xv @ w.T / (2 * len(xv)) ** 0.5, xa @ w.T / (2 * len(xa)) ** 0.5, reg.sqrt() * w[:, k].T])
    expected = torch.linalg.lstsq(design, target).solution.T
    torch.testing.assert_close(prior + delta, expected, rtol=1e-9, atol=1e-9)


def test_ffn_whole_heads_and_full_width_keep_everything():
    torch.manual_seed(7)
    xv, xa, w = torch.randn(80, 9), torch.randn(60, 9), torch.randn(4, 9)
    cols, heads, _ = ffn.choose(xv, xa, w, 3, head_dim=3)
    assert torch.equal(cols, torch.arange(9)) and torch.equal(heads, torch.arange(3))


def test_ffn_strength_never_beats_parent_on_video_budget():
    torch.manual_seed(0)
    xv, xa, w = torch.randn(400, 32), 0.2 * torch.randn(200, 32), torch.randn(16, 32)
    cols, _, _ = ffn.choose(xv, xa, w, 20, audio_weight=2.0)
    g, _ = ffn.gram(xv, xa, w, audio_weight=2.0)
    prior, delta, _ = ffn.fit(w, g, cols)
    _, report = ffn.choose_strength(xv, xa, w, cols, prior, delta)
    plain = next(c for c in report["candidates"] if c["alpha"] == 0.0)
    assert report["selected"]["audio_relmse"] <= plain["audio_relmse"] + 1e-12


def _time_embedder(freq=64, hidden=48, out=32):
    torch.manual_seed(1)
    return {"linear_1.weight": torch.randn(hidden, freq) * 0.1, "linear_1.bias": torch.randn(hidden) * 0.1,
            "linear_2.weight": torch.randn(out, hidden) * 0.1, "linear_2.bias": torch.randn(out) * 0.1}


def test_adaln_full_rank_is_exact_and_error_falls_with_rank():
    inputs = adaln.adaln_inputs(_time_embedder(), 64, adaln.default_timesteps())
    fac = adaln.AdaLNFactorizer(inputs)
    w, b = torch.randn(6 * 8 * 3, 32, dtype=torch.float64), torch.randn(6 * 8 * 3, dtype=torch.float64)
    errs = [fac.factorize(w, b, r, dtype=torch.float64)[1]["fp64_rel_l2"] for r in (1, 2, 4, fac.covariance_rank)]
    # Whitening drops directions below 1e-5 of the top singular value, so "full rank" leaves a ~1e-6 residual.
    assert errs == sorted(errs, reverse=True) and errs[-1] < 1e-5


def test_adaln_inputs_cover_schedule_and_reference_timestep():
    ts = adaln.default_timesteps()
    assert torch.isclose(ts, torch.tensor(adaln.REFERENCE_TIMESTEP)).any()
    for t in adaln.schedule_timesteps():
        assert torch.isclose(ts, torch.tensor(t)).any()


def test_blocks_protects_audio_and_drops_lowest_scores():
    layers = []
    for i in range(10):
        layers.append({"layer": i, **{m: dict(mean=0.0, p90=float(i if m != "a" else 9 - i), maximum=0.0)
                                     for m in blocks.MODALITIES}})
    plan = blocks.depth_candidates(layers, drop=2, protect=3, audio_weights=(1.0,))
    assert plan["protected_audio_layers"] == [0, 1, 2]
    assert all(not set(c["drop_layers"]) & {0, 1, 2} for c in plan["candidates"])


def test_blocks_remap_renumbers_contiguously():
    state = {f"transformer_blocks.{i}.w": torch.tensor([float(i)]) for i in range(5)} | {"proj_in.w": torch.zeros(1)}
    out = blocks.remap_blocks(state, [0, 2, 4])
    assert [out[f"transformer_blocks.{j}.w"].item() for j in range(3)] == [0.0, 2.0, 4.0] and "proj_in.w" in out
