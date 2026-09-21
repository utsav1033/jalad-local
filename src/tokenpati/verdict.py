"""daudega / chalega / ghare jake sutti babu."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .budget import Estimate

SLOW_TOK_S = 10.0      # below this chat feels broken
FAST_TOK_S = 18.0      # above this it reads faster than you do
DAUDEGA_MAX = 3


def is_fast(tok_s: float) -> bool:
    return round(tok_s) >= FAST_TOK_S


def is_usable(tok_s: float) -> bool:
    return round(tok_s) >= SLOW_TOK_S


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
    """Per model: highest precision that clears FAST; else the fastest quant that clears SLOW;
    else the smallest quant, so the table can show how far off it is."""
    by_model: dict[str, list[Estimate]] = {}
    for e in estimates:
        by_model.setdefault(e.model.name, []).append(e)
    out = []
    for group in by_model.values():
        group.sort(key=lambda e: e.weight_gib)          # ascending precision
        ok = [e for e in group if e.fits]
        fast = [e for e in ok if is_fast(e.tok_s)]
        usable = [e for e in ok if is_usable(e.tok_s)]
        if fast:
            out.append(fast[-1])
        elif usable:
            out.append(usable[0])
        else:
            out.append(group[0])
    return out


def rank(estimates: list[Estimate]) -> list[Ranked]:
    """Bigger model wins among those that run; daudega for the top 3 that are also fast."""
    runs = [e for e in estimates if e.fits and is_usable(e.tok_s)]
    dead = [e for e in estimates if e not in runs]
    runs.sort(key=lambda e: (e.model.params_b, e.tok_s), reverse=True)
    dead.sort(key=lambda e: e.model.params_b)

    ranked: list[Ranked] = []
    daudega_left = DAUDEGA_MAX
    for e in runs:
        if daudega_left and is_fast(e.tok_s):
            v = Verdict.DAUDEGA
            daudega_left -= 1
        else:
            v = Verdict.CHALEGA
        ranked.append(Ranked(e, v, 0))
    # daudega rows float to the top, then chalega, both by size; sutti at the bottom
    ranked.sort(key=lambda r: (r.verdict is not Verdict.DAUDEGA, -r.estimate.model.params_b))
    ranked += [Ranked(e, Verdict.SUTTI, 0) for e in dead]
    for i, r in enumerate(ranked, 1):
        r.rank = i
    return ranked


def winner(ranked: list[Ranked]) -> Ranked | None:
    for r in ranked:
        if r.verdict is not Verdict.SUTTI:
            return r
    return None
