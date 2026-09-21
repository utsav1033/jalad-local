"""The maths. This is the whole tool; everything else is presentation.

Three formulas. Write them, then test them against a real run.

1. Weight bytes
     weights = params * bytes_per_param(quant) * OVERHEAD
   bytes_per_param is roughly:
     Q4_K_M 0.56   Q5_K_M 0.68   Q6_K 0.80   Q8_0 1.06   FP16 2.0
   (the K-quants are above their nominal bits because some tensors stay at higher precision)
   OVERHEAD ~1.1 covers the embedding table, runtime buffers, and compute scratch.

2. KV cache bytes, for a given context length
     kv = 2 * layers * kv_heads * head_dim * context * bytes_per_element
   2 is for K and V. bytes_per_element is 2 for fp16 KV, 1 for q8 KV.
   This is the number everyone forgets. It grows linearly with context and is
   why a model that loads fine at 4k dies at 32k.

3. Predicted decode tok/s
     tok_s = bandwidth_bytes_per_s / (weights + kv_at_current_context) * EFFICIENCY
   Decode is memory-bound: every token re-reads all the weights once.
   EFFICIENCY is a fudge factor, expect 0.5 to 0.8 depending on backend. Start at 0.7,
   then measure and correct. That correction is the point of the project.

Fit rule
     fits = weights + kv_at_target_context + OS_RESERVE_GB < memory_gb
   OS_RESERVE_GB: ~4 on a Mac with a desktop running, ~1.5 headless Linux.
"""

from __future__ import annotations

from dataclasses import dataclass

from .catalog import Model
from .hardware import Hardware

BYTES_PER_PARAM: dict[str, float] = {
    "Q4_K_M": 0.56,
    "Q5_K_M": 0.68,
    "Q6_K": 0.80,
    "Q8_0": 1.06,
    "FP16": 2.0,
}

QUANTS_TO_TRY = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "FP16"]


@dataclass
class Estimate:
    model: Model
    quant: str
    weight_gb: float
    kv_gb: float            # at target_context
    total_gb: float         # weights + kv + os reserve
    fits: bool
    tok_s: float
    max_context: int        # largest context that still fits, capped at model.max_context


def weight_gb(model: Model, quant: str) -> float:
    raise NotImplementedError


def kv_gb(model: Model, context: int, kv_bytes: int = 2) -> float:
    raise NotImplementedError


def predicted_tok_s(hw: Hardware, weight_gb: float, kv_gb: float) -> float:
    raise NotImplementedError


def max_context_that_fits(hw: Hardware, model: Model, quant: str) -> int:
    """Largest context where weights + kv + reserve still fits. Binary search or just step through powers of two."""
    raise NotImplementedError


def estimate(hw: Hardware, model: Model, quant: str, target_context: int = 8192) -> Estimate:
    raise NotImplementedError


def estimate_all(hw: Hardware, catalog: list[Model], target_context: int = 8192) -> list[Estimate]:
    """One Estimate per (model, quant). Let verdict.py pick the best quant per model."""
    raise NotImplementedError
