# jalad-local

> **kaun banega tokenpati** — a game show where your machine sits in the hot seat and the models are the contestants.

Run one command. It reads your hardware, works out which local LLMs will actually run on it, at what quant, on which backend, and how many tokens per second you'll get. Then it hands you the exact command to run.

## Verdicts

| Verdict | Meaning | Rule |
|---|---|---|
| `daudega` | it'll sprint | top 3 by predicted tok/s, and only if above the fast threshold |
| `chalega` | it'll do | fits in memory with headroom, usable speed |
| `ghare jake sutti babu` | go home and sleep | doesn't fit, or fits but unusably slow |

## Install

```sh
pip install tokenpati
tokenpati
```

Or without installing anything permanently:

```sh
uvx tokenpati
```

Straight from the repo:

```sh
pip install git+https://github.com/utsav1033/jalad-local
```

## Develop

```sh
git clone https://github.com/utsav1033/jalad-local && cd jalad-local
uv sync
uv run tokenpati
uv run pytest
```

## Release

Bump `version` in `pyproject.toml`, then tag it. GitHub Actions builds and publishes to PyPI.

```sh
git tag v0.1.0 && git push --tags
```

Flags: `--context 32768` budgets memory for a longer context, `--fast` skips the game-show pause, `--json` for scripts.

## How it decides

Every number is derived from four things about your machine: memory, memory bandwidth, backend, and how much of that memory the GPU may actually use (Apple caps it at about 75% of unified memory).

- **Weights** = params × bytes per param at that quant × 1.1 overhead.
- **KV cache** = 2 × layers × kv heads × head dim × context × 2 bytes. Budgeted at the context you ask for.
- **tok/s** = bandwidth ÷ (active weights + KV at 2k) × 0.8. Decode is memory-bound, so this one line predicts speed for any model on any machine. MoE models use their active parameters, which is why a 30B-A3B can outrun an 8B.
- **Quant** = the highest precision that still clears the fast bar, else the fastest one that clears the slow bar.
- **Winner** = the biggest model that gets `daudega`.

Every tok/s is a prediction. Run the model, measure, and adjust `EFFICIENCY` in `budget.py` if it's off.

## Phase 1 (now): analyzer only

Phase 2 adds `tokenpati serve`, which picks the backend and runs the winner.

## Layout

```
src/tokenpati/
  hardware.py   what machine is this: chip, RAM, bandwidth, backend
  catalog.py    the contestants: model families and their architecture numbers
  budget.py     the maths: weight bytes, KV cache bytes, predicted tok/s
  verdict.py    daudega / chalega / ghare jake sutti babu
  ui.py         the game show: banner, gauges, table, final command
  cli.py        entry point
docs/wireframe.txt   the screen, sketched before any UI code
tests/               predict, measure, correct
```

## The rule

Predict, measure, correct. Every tok/s number this tool prints is a prediction. Run the model, measure the real number, fix the formula.
