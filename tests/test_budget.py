"""Predict, measure, correct. Units are GiB."""

import pytest

from tokenpati import budget, verdict
from tokenpati.catalog import CATALOG, Model
from tokenpati.hardware import Hardware

LLAMA_8B = next(m for m in CATALOG if m.name == "Llama 3.1 8B")


def test_kv_cache_llama_8b_at_8k_is_one_gib():
    # 2 * 32 * 8 * 128 * 8192 * 2 bytes = 1 GiB exactly
    assert budget.kv_gib(LLAMA_8B, 8192) == pytest.approx(1.0)


def test_weight_q4_llama_8b():
    # 8.03e9 * 0.56 * 1.1 / 2^30 = 4.61 GiB; the real Q4_K_M GGUF is 4.58 GiB
    assert budget.weight_gib(LLAMA_8B, "Q4_K_M") == pytest.approx(4.6, abs=0.15)


def test_m4_16gb_picks_something_sane():
    hw = Hardware(chip="Apple M4", os="darwin", memory_gib=16, usable_gib=12, bandwidth_gbps=120,
                  bandwidth_estimated=False, backend="mlx", gpu_cores=10)
    ranked = verdict.rank(verdict.best_quant_per_model(budget.estimate_all(hw, CATALOG)))
    win = verdict.winner(ranked)
    assert win is not None
    assert win.estimate.fits
    assert win.estimate.model.params_b <= 15
    assert any(r.verdict is verdict.Verdict.SUTTI for r in ranked)


@pytest.mark.skip(reason="fill in from a real run on your machine, then unskip")
def test_prediction_matches_measurement():
    hw = Hardware(chip="?", os="darwin", memory_gib=0, usable_gib=0, bandwidth_gbps=0,
                  bandwidth_estimated=False, backend="mlx")
    measured_tok_s = 0.0
    est = budget.estimate(hw, LLAMA_8B, "Q4_K_M", target_context=8192)
    assert est.tok_s == pytest.approx(measured_tok_s, rel=0.25)
