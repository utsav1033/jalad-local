"""tokenpati: read the machine, run the show, print the answer."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from rich.console import Console

from . import budget, catalog, hardware, ui, verdict


def main() -> None:
    parser = argparse.ArgumentParser(prog="tokenpati", description="kaun banega tokenpati")
    parser.add_argument("--context", type=int, default=8192, help="context length to budget for (default 8192)")
    parser.add_argument("--fast", action="store_true", help="skip the dramatic pause")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    console = Console()
    hw = hardware.detect()
    estimates = budget.estimate_all(hw, catalog.CATALOG, target_context=args.context)
    ranked = verdict.rank(verdict.best_quant_per_model(estimates))
    win = verdict.winner(ranked)

    if args.json:
        print(json.dumps({
            "hardware": asdict(hw),
            "winner": win.estimate.model.name if win else None,
            "ranked": [{
                "rank": r.rank, "model": r.estimate.model.name, "quant": r.estimate.quant,
                "weight_gib": round(r.estimate.weight_gib, 2), "kv_gib": round(r.estimate.kv_gib, 2),
                "fits": r.estimate.fits, "tok_s": round(r.estimate.tok_s, 1),
                "max_context": r.estimate.max_context, "verdict": r.verdict.value,
            } for r in ranked],
        }, indent=2))
        return

    ui.banner(console)
    ui.wait_for_it(console, fast=args.fast)
    ui.hot_seat(console, hw)
    console.print()
    ui.leaderboard(console, hw, ranked)
    console.print()
    if win:
        ui.final_answer(console, hw, win)
    else:
        ui.nobody_wins(console)
    console.print()


if __name__ == "__main__":
    main()
