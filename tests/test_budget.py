"""Predict, measure, correct.

The first two tests pin the formulas to numbers you can check by hand.
The third is the one that matters: fill in a real measurement from your machine
and make the prediction land within tolerance. If it doesn't, fix EFFICIENCY or
OVERHEAD in budget.py, not the test.
"""

import pytest

from tokenpati import budget
from tokenpati.catalog import Model
from tokenpati.hardware import Hardware

LLAMA_8B = Model("Llama 3.1 8B", 8.0, 32, 8, 128, 131072, "meta-llama/Llama-3.1-8B-Instruct")


def test_kv_cache_llama_8b_at_8k():
    # 2 * 32 layers * 8 kv_heads * 128 head_dim * 8192 ctx * 2 bytes = 1,073,741,824 bytes = 1 GiB
    assert budget.kv_gb(LLAMA_8B, 8192) == pytest.approx(1.0, rel=0.01)


def test_weight_q4_llama_8b_is_roughly_4_9_gb():
    # 8.0B * 0.56 * 1.1 overhead ~= 4.93 GB. Real Q4_K_M GGUF is 4.92 GB.
    assert budget.weight_gb(LLAMA_8B, "Q4_K_M") == pytest.approx(4.9, abs=0.3)


@pytest.mark.skip(reason="fill in from a real run on your machine, then unskip")
def test_prediction_matches_measurement():
    hw = Hardware(chip="?", os="darwin", memory_gb=0, bandwidth_gbps=0, backend="mlx")
    measured_tok_s = 0.0
    est = budget.estimate(hw, LLAMA_8B, "Q4_K_M", target_context=8192)
    assert est.tok_s == pytest.approx(measured_tok_s, rel=0.25)
