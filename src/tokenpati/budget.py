"""The maths. Units: GiB everywhere, GB/s for bandwidth.

weights = params * bytes_per_param(quant) * OVERHEAD
kv      = 2 * layers * kv_heads * head_dim * context * kv_bytes
tok/s   = bandwidth / (active_weights + kv(SPEED_CONTEXT)) * EFFICIENCY
          decode is memory-bound: every token re-reads the active weights plus whatever KV is filled.
          Speed is quoted at SPEED_CONTEXT (a typical chat), memory is budgeted at target_context (the worst case).
fits    = weights + kv(target_context) <= hw.usable_gib
"""

from __future__ import annotations

from dataclasses import dataclass

from .catalog import Model
from .hardware import Hardware

GIB = 1024**3

BYTES_PER_PARAM: dict[str, float] = {
    "Q4_K_M": 0.56,
    "Q5_K_M": 0.68,
    "Q6_K": 0.80,
    "Q8_0": 1.06,
    "FP16": 2.0,
}
QUANTS = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "FP16"]
QUANT_LABEL = {"Q4_K_M": "4-bit", "Q5_K_M": "5-bit", "Q6_K": "6-bit", "Q8_0": "8-bit", "FP16": "fp16"}
MLX_QUANT = {"Q4_K_M": "4bit", "Q6_K": "6bit", "Q8_0": "8bit", "FP16": "bf16"}   # no 5-bit on mlx

OVERHEAD = 1.10          # embeddings, runtime buffers, scratch
EFFICIENCY = 0.80        # fraction of peak bandwidth real decode achieves; measure and correct
KV_BYTES = 2             # fp16 KV cache
SPEED_CONTEXT = 2048     # context fill assumed when quoting tok/s


@dataclass
class Estimate:
    model: Model
    quant: str
    weight_gib: float
    kv_gib: float           # at target context
    total_gib: float
    fits: bool
    tok_s: float
    max_context: int
    target_context: int

    @property
    def headroom_gib(self) -> float:
        return self.total_gib


def weight_gib(model: Model, quant: str) -> float:
    return model.params_b * 1e9 * BYTES_PER_PARAM[quant] * OVERHEAD / GIB


def active_weight_gib(model: Model, quant: str) -> float:
    return model.active_b * 1e9 * BYTES_PER_PARAM[quant] * OVERHEAD / GIB


def kv_gib(model: Model, context: int, kv_bytes: int = KV_BYTES) -> float:
    return 2 * model.layers * model.kv_heads * model.head_dim * context * kv_bytes / GIB


def predicted_tok_s(hw: Hardware, active_gib: float, kv: float) -> float:
    bytes_per_token = (active_gib + kv) * GIB
    return hw.bandwidth_gbps * 1e9 / bytes_per_token * EFFICIENCY


def max_context_that_fits(hw: Hardware, model: Model, quant: str) -> int:
    w = weight_gib(model, quant)
    best = 0
    ctx = 1024
    while ctx <= model.max_context:
        if w + kv_gib(model, ctx) <= hw.usable_gib:
            best = ctx
        ctx *= 2
    return best


def estimate(hw: Hardware, model: Model, quant: str, target_context: int = 8192) -> Estimate:
    w = weight_gib(model, quant)
    kv = kv_gib(model, target_context)
    total = w + kv
    return Estimate(
        model=model, quant=quant, weight_gib=w, kv_gib=kv, total_gib=total,
        fits=total <= hw.usable_gib,
        tok_s=predicted_tok_s(hw, active_weight_gib(model, quant), kv_gib(model, min(SPEED_CONTEXT, target_context))),
        max_context=min(max_context_that_fits(hw, model, quant), model.max_context),
        target_context=target_context,
    )


def estimate_all(hw: Hardware, catalog: list[Model], target_context: int = 8192) -> list[Estimate]:
    quants = [q for q in QUANTS if not hw.is_apple or q in MLX_QUANT]
    return [estimate(hw, m, q, target_context) for m in catalog for q in quants]
