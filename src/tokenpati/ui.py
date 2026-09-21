"""The game show. Layout is in docs/wireframe.txt; build to that, top to bottom.

Colour rule: four colours and a dim, each with one meaning.
  green   daudega, fits with room
  amber   chalega, fits but tight or slow
  red     sutti, doesn't fit
  accent  headings, the final command box
  dim     labels, headroom, anything not a decision

Sections (one function each, all take a rich Console)
  banner        pyfiglet "KAUN BANEGA TOKENPATI", gradient via rich Text.stylize per char
  hot_seat      the hardware card: chip, RAM, bandwidth, backend
  memory_bar    per model: [weights][kv][free] stacked bar, three colours, one line
  speedometer   tok/s gauge, coloured by SLOW/FAST thresholds
  leaderboard   the ranked table with the verdict column
  final_answer  bordered panel with the exact run command, brightest thing on screen
"""

from __future__ import annotations

from rich.console import Console

from .hardware import Hardware
from .verdict import Ranked


def banner(console: Console) -> None:
    raise NotImplementedError


def hot_seat(console: Console, hw: Hardware) -> None:
    raise NotImplementedError


def memory_bar(hw: Hardware, weight_gb: float, kv_gb: float, width: int = 40) -> str:
    """Return a one-line bar; leaderboard() embeds it in a column."""
    raise NotImplementedError


def leaderboard(console: Console, hw: Hardware, ranked: list[Ranked]) -> None:
    raise NotImplementedError


def final_answer(console: Console, hw: Hardware, winner: Ranked) -> None:
    raise NotImplementedError
