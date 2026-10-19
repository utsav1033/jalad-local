# jalad-local

> **kaun banega tokenpati**: your machine sits in the hot seat, the models are the contestants, and the winner actually gets to play.

`tokenpati` reads your hardware, works out which local LLMs will run on it (at what quant, on which engine, at how many tokens per second), and then **downloads, loads and chats with the one you pick**, all from one terminal app. Think LM Studio, in your terminal, open source, and honest about the maths.

Single static binary. No Python, no pip, no venv.

## Install

```sh
curl -fsSL https://raw.githubusercontent.com/utsav1033/jalad-local/main/install.sh | sh
tokenpati
```

Or grab a binary from [Releases](https://github.com/utsav1033/jalad-local/releases), or build from source with `cargo install --git https://github.com/utsav1033/jalad-local`.

You do not need any engine installed first. If nothing is there, tokenpati downloads the official `llama-server` build for your OS and GPU (CPU, CUDA, Vulkan, ROCm or Metal) into `~/.tokenpati/bin`, with no sudo.

## The app

```
 1 Machine   2 Models   3 Search   4 Library   5 Launch   6 Chat   7 Setup
```

| Tab | What you do there |
|---|---|
| **Machine** | See the hot seat: chip, GPU, memory, usable memory, bandwidth, live RAM/CPU/GPU. Press `b` to *measure* memory bandwidth instead of guessing it. Your best bets are listed; `enter` launches the winner. |
| **Models** | The leaderboard. 21 hand-picked models, each with a memory bar (weights, KV cache, free), predicted tok/s, your *measured* tok/s, max context, verdict. `←→` changes quant, `+/-` changes the context budget, and everything recomputes as you go. The detail pane shows the whole quant matrix and *why* it got its verdict. |
| **Search** | Anything not in the Models list: search Hugging Face (where people publish open models), pick a size, and see if it fits before downloading. tokenpati reads just the first few MB of the file to get the model's real shape, so the numbers hold for models it has never heard of. |
| **Library** | Everything on disk: ollama's models, GGUFs tokenpati downloaded, and LM Studio's model folder. Launch, unload or delete (it asks first). |
| **Launch** | Every launch is five visible steps, each a block that shows progress, logs and timing: **Engine → Weights → Load → Warm-up → Ready**. Downloads show speed and ETA and resume if cut off. Cancel any time, retry with `r`. |
| **Chat** | Streaming chat with the loaded model. Thinking models show their reasoning in a dim block (`/think` toggles it). Every reply shows tok/s, tokens and time to first token. Context use is a live gauge. `/system`, `/temp`, `/copy`, `/save`, `/clear`. |
| **Setup** | ollama, llama.cpp and mlx: installed? running? which version? `i` installs, `s` starts ollama, `←→` picks the llama.cpp build, `enter` marks your preferred engine. |

Nothing downloads or starts until you have seen the **launch plan**: quant, engine, context and GPU layers are all adjustable there, with memory, speed, download size, free disk and the exact command recomputed live.

Once a model is ready it is served on an **OpenAI-compatible endpoint** (`http://127.0.0.1:8686/v1` for llama.cpp, ollama's own port for ollama), so your editor, scripts and agents can use it while you chat. `y` copies the URL.

Keys: `1-7`/`tab` switch tabs (`alt+1-7` works even while typing), `?` help, `,` settings, `L` log panel, `q` quit. Quitting with a model loaded asks whether to stop it or leave it serving.

## Without the app

```sh
tokenpati show                 # the original one-shot game show
tokenpati --json               # machine-readable ranking, for scripts
tokenpati serve                # download + load the winner, serve it, ctrl-c to stop
tokenpati serve "qwen3 8b" --engine llama.cpp --quant Q5_K_M --port 9000
```

`--context 32768` budgets for a longer context everywhere. When stdout is not a terminal, `tokenpati` prints the game show instead of opening the app.

## Engines

| Engine | Best for | How tokenpati uses it |
|---|---|---|
| **ollama** | getting started | `ollama pull`, load with your context size, chat over its API. If ollama has no build at the quant you picked, tokenpati says so and pulls the same quant from Hugging Face (`hf.co/<repo>:<quant>`). |
| **llama.cpp** | control | tokenpati installs `llama-server`, downloads the exact GGUF from Hugging Face, and runs it with your context and `-ngl` GPU layer count. Partial offload lets models bigger than VRAM run with some layers in RAM. |
| **mlx** | Apple silicon | downloads the `mlx-community` weights and runs `mlx_lm.server`. |

## Verdicts

| Verdict | Rule |
|---|---|
| **runs fast** | top 3 by size among models that clear the fast bar (18 tok/s, faster than you read) |
| **runs** | fits in memory at a usable speed (10+ tok/s) |
| **too big** / **too slow** | doesn't fit, or fits but slower than you read |

You can still launch a "too slow" model. The plan tells you what will happen; it's your machine. `--json` keeps the original codes (`daudega`, `chalega`, `ghare jake sutti babu`) so existing scripts keep working.

## How it decides

Every number comes from four things about your machine: memory, memory bandwidth, backend, and how much of that memory the GPU may actually use (Apple caps it at about 75% of unified memory).

- **Weights** = params × bytes per param at that quant × 1.1 overhead.
- **KV cache** = 2 × layers × kv heads × head dim × context × 2 bytes. Budgeted at the context you ask for.
- **tok/s** = bandwidth ÷ (active weights + KV at 2k) × efficiency. Decode is memory-bound, so this one line predicts speed for any model on any machine. MoE models only read their active parameters, which is why a 30B-A3B can outrun an 8B.
- **Partial offload** (discrete GPUs): layers that don't fit in VRAM sit in RAM, and time per token is the GPU read plus the RAM read.
- **Quant** = the highest precision that still clears the fast bar, else the fastest one that clears the slow bar.
- **Winner** = the biggest model that **runs fast**.

Every tok/s starts as a prediction. Each time a model runs, tokenpati measures the real speed and calibrates its efficiency factor for your machine, so predictions get better the more you use it. The Models tab shows predicted and measured side by side.

## Files

Everything lives in `~/.tokenpati` (override with `TOKENPATI_HOME`): `models/`, `bin/` (llama.cpp), `logs/` (server output), `chats/` (`/save`), and `state.json` (settings and measurements). Delete the folder and it's all gone.

Environment: `HF_TOKEN` for gated Hugging Face repos, `HF_ENDPOINT` for a mirror, `OLLAMA_HOST` for a non-default ollama, `HTTPS_PROXY`/`NO_PROXY` are honoured.

## Develop

```sh
git clone https://github.com/utsav1033/jalad-local && cd jalad-local
cargo run --release
cargo test
```

Layout:

- `hardware.rs` detects the machine, measures bandwidth, samples live stats
- `catalog.rs` lists the contestants
- `budget.rs` does the maths (`verdict.rs` hands out verdicts)
- `gguf.rs` reads GGUF headers, `hf.rs` talks to Hugging Face, `net.rs` downloads (resumable, proxy-aware)
- `engine/` drives ollama, llama.cpp and mlx, and streams chat
- `pipeline.rs` is the five-step launch
- `tui/` is the app; `show.rs` is the classic game show

## Release

Bump `version` in `Cargo.toml`, then tag. GitHub Actions builds binaries for macOS (arm64, x86_64) and Linux (x86_64, arm64) and attaches them to the release.

```sh
git tag v0.3.0 && git push --tags
```

The old Python version lives on PyPI as `tokenpati` 0.1.0 and is no longer maintained.
