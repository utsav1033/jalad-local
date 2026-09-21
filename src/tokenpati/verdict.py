"""daudega / chalega / ghare jake sutti babu.

Rules
  ghare jake sutti babu   doesn't fit at any quant, OR fits but tok_s < SLOW_TOK_S
  chalega                 fits, tok_s >= SLOW_TOK_S
  daudega                 chalega AND tok_s >= FAST_TOK_S AND in the top DAUDEGA_MAX by tok_s

Per model, keep only the best quant: the highest-precision quant that still fits
at the target context with tok_s above SLOW_TOK_S. Bigger quant beats faster quant
as long as it clears the bar, because quality is what people actually want.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .budget import Estimate

SLOW_TOK_S = 8.0      # below this, chat feels broken
FAST_TOK_S = 30.0     # above this, it feels instant
DAUDEGA_MAX = 3


class Verdict(str, Enum):
    DAUDEGA = "daudega"
    CHALEGA = "chalega"
    SUTTI = "ghare jake sutti babu"


@dataclass
class Ranked:
    estimate: Estimate
    verdict: Verdict
    rank: int


def best_quant_per_model(estimates: list[Estimate]) -> list[Estimate]:
    raise NotImplementedError


def rank(estimates: list[Estimate]) -> list[Ranked]:
    """Sorted: daudega first (by tok_s desc), then chalega, then sutti."""
    raise NotImplementedError
