# jalad-local

> **kaun banega tokenpati** — a game show where your machine sits in the hot seat and the models are the contestants.

Run one command. It reads your hardware, works out which local LLMs will actually run on it, at what quant, on which backend, and how many tokens per second you'll get. Then it hands you the exact command to run.

Single static binary. No Python, no pip, no venv.

## Install

```sh
curl -fsSL https://raw.githubusercontent.com/utsav1033/jalad-local/main/install.sh | sh
tokenpati
```

Or grab a binary from [Releases](https://github.com/utsav1033/jalad-local/releases), or build from source with `cargo install --git https://github.com/utsav1033/jalad-local`.

Flags: `--context 32768` budgets memory for a longer context, `--fast` skips the game-show pause, `--json` for scripts.

## Verdicts

| Verdict | Meaning | Rule |
|---|---|---|
| `daudega` | it'll sprint | top 3 by size among models that clear the fast bar |
| `chalega` | it'll do | fits in memory, usable speed |
| `ghare jake sutti babu` | go home and sleep | doesn't fit, or fits but unusably slow |

## How it decides

Every number is derived from four things about your machine: memory, memory bandwidth, backend, and how much of that memory the GPU may actually use (Apple caps it at about 75% of unified memory).

- **Weights** = params × bytes per param at that quant × 1.1 overhead.
- **KV cache** = 2 × layers × kv heads × head dim × context × 2 bytes. Budgeted at the context you ask for.
- **tok/s** = bandwidth ÷ (active weights + KV at 2k) × 0.8. Decode is memory-bound, so this one line predicts speed for any model on any machine. MoE models use their active parameters, which is why a 30B-A3B can outrun an 8B.
- **Quant** = the highest precision that still clears the fast bar, else the fastest one that clears the slow bar.
- **Winner** = the biggest model that gets `daudega`.

Every tok/s is a prediction. Run the model, measure, and adjust `EFFICIENCY` in `src/budget.rs` if it's off.

## Develop

```sh
git clone https://github.com/utsav1033/jalad-local && cd jalad-local
cargo run --release
cargo test
```

Layout: `hardware.rs` detects the machine, `catalog.rs` lists the contestants, `budget.rs` is the maths, `verdict.rs` hands out verdicts, `ui.rs` runs the show. Figlet fonts are embedded from `fonts/`.

## Release

Bump `version` in `Cargo.toml`, then tag. GitHub Actions builds binaries for macOS (arm64, x86_64) and Linux (x86_64, arm64) and attaches them to the release.

```sh
git tag v0.2.0 && git push --tags
```

Phase 2 will add `tokenpati serve`, which picks the backend and runs the winner.

The old Python version lives on PyPI as `tokenpati` 0.1.0 and is no longer maintained.
