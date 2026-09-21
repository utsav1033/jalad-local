# jalad-local

> **kaun banega tokenpati** — a game show where your machine sits in the hot seat and the models are the contestants.

Run one command. It reads your hardware, works out which local LLMs will actually run on it, at what quant, on which backend, and how many tokens per second you'll get. Then it hands you the exact command to run.

## Verdicts

| Verdict | Meaning | Rule |
|---|---|---|
| `daudega` | it'll sprint | top 3 by predicted tok/s, and only if above the fast threshold |
| `chalega` | it'll do | fits in memory with headroom, usable speed |
| `ghare jake sutti babu` | go home and sleep | doesn't fit, or fits but unusably slow |

## Phase 1 (this repo, now): analyzer only

Detect hardware, compute budgets, print the verdict table. No serving yet.

## Phase 2 (later): serve

`tokenpati serve <model>` picks the backend and runs it.

## Run

```sh
uv sync
uv run tokenpati
```

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
