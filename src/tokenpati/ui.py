"""The game show."""

from __future__ import annotations

import time

import pyfiglet
from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from .budget import MLX_QUANT, QUANT_LABEL
from .hardware import Hardware
from .verdict import Ranked, Verdict, is_fast, is_usable

GREEN = "#22c55e"
AMBER = "#f59e0b"
RED = "#ef4444"
ACCENT = "#ff9933"       # saffron
DIM = "grey50"

SAFFRON = (255, 153, 51)
WHITE = (255, 255, 255)
INDIA_GREEN = (19, 136, 8)

VERDICT_STYLE = {
    Verdict.DAUDEGA: f"bold {GREEN}",
    Verdict.CHALEGA: AMBER,
    Verdict.SUTTI: RED,
}

STAGES = [
    "scanning the hot seat",
    "inviting the contestants",
    "weighing the weights",
    "counting the KV cache",
    "phoning a friend",
    "audience poll",
    "computerji, lock kiya jaye",
]


def _lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _gradient_text(s: str) -> Text:
    """Saffron -> white -> green across each line, tiranga style."""
    out = Text()
    for line in s.splitlines():
        n = max(len(line) - 1, 1)
        for i, ch in enumerate(line):
            t = i / n
            rgb = _lerp(SAFFRON, WHITE, t * 2) if t < 0.5 else _lerp(WHITE, INDIA_GREEN, (t - 0.5) * 2)
            out.append(ch, style=f"bold rgb({rgb[0]},{rgb[1]},{rgb[2]})")
        out.append("\n")
    return out


def _fig(text: str, font: str) -> str:
    lines = pyfiglet.Figlet(font=font, width=400).renderText(text).splitlines()
    return "\n".join(l for l in lines if l.strip())


def banner(console: Console) -> None:
    w = console.width
    if w >= 116:
        top, big = _fig("KAUN BANEGA", "small"), _fig("TOKENPATI", "dos_rebel")
    elif w >= 78:
        top, big = _fig("KAUN BANEGA", "small"), _fig("TOKENPATI", "ansi_regular")
    else:
        top, big = "K A U N   B A N E G A", _fig("TOKENPATI", "small")
    console.print()
    console.print(Align.center(Text(top, style=f"bold {ACCENT}")))
    console.print(Align.center(_gradient_text(big)))
    console.print(Align.center(Text("which local model will actually run on this thing", style=f"italic {DIM}")))
    console.print()


def wait_for_it(console: Console, fast: bool = False) -> None:
    """The dramatic pause. Analysis itself is instant."""
    with Progress(
        SpinnerColumn("dots", style=ACCENT),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=30, style=DIM, complete_style=ACCENT, finished_style=GREEN),
        TextColumn(f"[{DIM}]{{task.percentage:>3.0f}}%"),
        console=console, transient=True,
    ) as p:
        task = p.add_task(STAGES[0], total=len(STAGES))
        for stage in STAGES:
            p.update(task, description=stage)
            time.sleep(0 if fast else 0.38)
            p.advance(task)
        time.sleep(0 if fast else 0.25)


def hot_seat(console: Console, hw: Hardware) -> None:
    grid = Table.grid(padding=(0, 3))
    grid.add_column(style=DIM, justify="right")
    grid.add_column(style="bold")
    grid.add_column(style=DIM, justify="right")
    grid.add_column(style="bold")
    bw = f"{hw.bandwidth_gbps:.0f} GB/s" + ("  (guess)" if hw.bandwidth_estimated else "")
    left = hw.gpu_name or hw.chip
    cores = f"{hw.gpu_cores} gpu cores" if hw.gpu_cores else hw.os
    grid.add_row("chip", left, "backend", hw.backend)
    grid.add_row("memory", hw.memory_label, "bandwidth", bw)
    grid.add_row("usable", f"{hw.usable_gib:.1f} GiB for models", "", cores)
    console.print(Panel(grid, title=f"[bold {ACCENT}]HOT SEAT", border_style=ACCENT, box=box.ROUNDED, padding=(0, 2)))


def memory_bar(hw: Hardware, weight: float, kv: float, fits: bool, width: int) -> Text:
    total = hw.usable_gib
    if not fits:
        over = min(width, int(round((weight + kv) / total * width)))
        t = Text("█" * width, style=RED)
        t.append(f" {weight + kv:.0f}G", style=RED)
        return t
    w = int(round(weight / total * width))
    k = int(round(kv / total * width))
    w, k = min(w, width), min(k, width - w)
    t = Text()
    t.append("█" * w, style=GREEN)
    t.append("█" * k, style=AMBER)
    t.append("░" * (width - w - k), style=DIM)
    return t


def _tok_style(tok_s: float) -> str:
    if is_fast(tok_s):
        return f"bold {GREEN}"
    if is_usable(tok_s):
        return AMBER
    return RED


def _ctx(n: int) -> str:
    return f"{n // 1024}k" if n >= 1024 else str(n)


def leaderboard(console: Console, hw: Hardware, ranked: list[Ranked]) -> None:
    bar_w = 24 if console.width >= 110 else 14
    table = Table(box=box.SIMPLE_HEAD, header_style=f"bold {ACCENT}", padding=(0, 1), expand=False)
    table.add_column("#", style=DIM, justify="right")
    table.add_column("contestant", style="bold")
    table.add_column("quant")
    table.add_column("memory", justify="right")
    table.add_column(f"[{GREEN}]weights[/] [{AMBER}]kv[/] [{DIM}]free[/]")
    table.add_column("tok/s", justify="right")
    table.add_column("ctx", justify="right", style=DIM)
    table.add_column("verdict")
    for r in ranked:
        e = r.estimate
        dead = r.verdict is Verdict.SUTTI
        row_style = DIM if dead else ""
        name = Text(e.model.name)
        if e.model.is_moe:
            name.append("  moe", style=f"italic {DIM}")
        table.add_row(
            str(r.rank),
            name,
            Text(QUANT_LABEL[e.quant], style=DIM if dead else ""),
            f"{e.total_gib:.1f} GiB",
            memory_bar(hw, e.weight_gib, e.kv_gib, e.fits, bar_w),
            Text("-" if dead else f"{e.tok_s:.0f}", style=RED if dead else _tok_style(e.tok_s)),
            "-" if dead else _ctx(e.max_context),
            Text(r.verdict.value, style=VERDICT_STYLE[r.verdict]),
            style=row_style,
        )
    ctx = ranked[0].estimate.target_context if ranked else 0
    console.print(Text(f"  budgeting for {ctx:,} tokens of context   (--context to change)", style=DIM))
    console.print(table)


def run_commands(hw: Hardware, r: Ranked) -> list[tuple[str, str]]:
    e = r.estimate
    quant_tag = e.quant[0].lower() + e.quant[1:]              # Q4_K_M -> q4_K_M
    cmds = []
    if hw.is_apple:
        repo = f"mlx-community/{e.model.mlx}-{MLX_QUANT[e.quant]}"
        cmds.append(("fastest on apple silicon", f"pip install mlx-lm && mlx_lm.chat --model {repo}"))
    cmds.append(("easy mode", f"ollama run {e.model.ollama}-{quant_tag}"))
    return cmds


def final_answer(console: Console, hw: Hardware, r: Ranked) -> None:
    e = r.estimate
    head = Text()
    head.append(e.model.name, style=f"bold {GREEN}")
    head.append("  ·  ", style=DIM)
    head.append(QUANT_LABEL[e.quant], style="bold")
    head.append("  ·  ", style=DIM)
    head.append(hw.backend, style="bold")
    head.append("  ·  ", style=DIM)
    head.append(f"~{e.tok_s:.0f} tok/s", style=_tok_style(e.tok_s))
    head.append("  ·  ", style=DIM)
    head.append(f"up to {_ctx(e.max_context)} context", style="bold")
    sub = Text(f"{e.model.tagline}.  {e.total_gib:.1f} GiB of your {hw.usable_gib:.1f} GiB.", style=f"italic {DIM}")

    lines = [head, sub, Text()]
    for label, cmd in run_commands(hw, r):
        lines.append(Text(f"{label}", style=DIM))
        lines.append(Text(f"  {cmd}", style=f"bold {ACCENT}"))
        lines.append(Text())
    lines.pop()
    console.print(Panel(Group(*lines), title=f"[bold {ACCENT}]FINAL ANSWER", subtitle=f"[{DIM}]{r.verdict.value}",
                        border_style=ACCENT, box=box.DOUBLE, padding=(1, 3)))


def nobody_wins(console: Console) -> None:
    console.print(Panel(Text("ghare jake sutti babu. nothing fits. try --context 2048, or a smaller machine budget.", style=RED),
                        border_style=RED, box=box.DOUBLE, padding=(1, 3)))
