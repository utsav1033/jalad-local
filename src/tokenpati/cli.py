"""tokenpati: read the machine, run the show, print the answer."""

from __future__ import annotations

import argparse

from rich.console import Console

from . import budget, catalog, hardware, ui, verdict


def main() -> None:
    parser = argparse.ArgumentParser(prog="tokenpati", description="kaun banega tokenpati")
    parser.add_argument("--context", type=int, default=8192, help="target context length to budget for")
    parser.add_argument("--json", action="store_true", help="print estimates as JSON instead of the show")
    args = parser.parse_args()

    console = Console()
    hw = hardware.detect()
    estimates = budget.estimate_all(hw, catalog.CATALOG, target_context=args.context)
    ranked = verdict.rank(verdict.best_quant_per_model(estimates))

    if args.json:
        raise NotImplementedError

    ui.banner(console)
    ui.hot_seat(console, hw)
    ui.leaderboard(console, hw, ranked)
    if ranked and ranked[0].verdict is not verdict.Verdict.SUTTI:
        ui.final_answer(console, hw, ranked[0])


if __name__ == "__main__":
    main()
