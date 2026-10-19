//! tokenpati: read the machine, rank the models, then download, load and chat with one.

mod budget;
mod catalog;
mod engine;
mod gguf;
mod hardware;
mod hf;
mod net;
mod pipeline;
mod show;
mod state;
mod tui;
mod verdict;

use clap::{Parser, Subcommand};
use serde::Serialize;
use std::io::IsTerminal;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// kaun banega tokenpati: which local LLM will actually run on this machine, and then run it
#[derive(Parser)]
#[command(name = "tokenpati", version)]
struct Args {
    #[command(subcommand)]
    cmd: Option<Cmd>,
    /// Context length to budget memory for
    #[arg(long, global = true)]
    context: Option<u32>,
    /// Skip the dramatic pause (game show) and print instead of opening the app
    #[arg(long, global = true)]
    fast: bool,
    /// Machine-readable output
    #[arg(long, global = true)]
    json: bool,
}

#[derive(Subcommand)]
enum Cmd {
    /// The original one-shot game show: hardware, leaderboard, final answer
    Show,
    /// Download (if needed), load and serve a model without the UI. Ctrl-C stops it.
    Serve {
        /// Catalog model, e.g. "qwen3 8b". Defaults to the winner.
        model: Option<String>,
        /// ollama | llama.cpp | mlx
        #[arg(long)]
        engine: Option<String>,
        /// Q4_K_M | Q5_K_M | Q6_K | Q8_0 | FP16
        #[arg(long)]
        quant: Option<String>,
        #[arg(long)]
        port: Option<u16>,
    },
}

#[derive(Serialize)]
struct Row<'a> {
    rank: usize,
    model: &'a str,
    quant: &'a str,
    weight_gib: f64,
    kv_gib: f64,
    fits: bool,
    offload: bool,
    tok_s: f64,
    max_context: u32,
    verdict: &'a str,
}

#[derive(Serialize)]
struct Report<'a> {
    hardware: &'a hardware::Hardware,
    winner: Option<&'a str>,
    ranked: Vec<Row<'a>>,
}

fn main() {
    let args = Args::parse();
    let st = state::State::load();
    let mut hw = hardware::detect();
    hw.apply_state(&st);
    let context = args.context.unwrap_or(st.settings.context);

    match args.cmd {
        Some(Cmd::Serve {
            model,
            engine,
            quant,
            port,
        }) => {
            std::process::exit(serve(&hw, &st, context, model, engine, quant, port));
        }
        Some(Cmd::Show) => return show_report(&hw, context, args.fast),
        None => {}
    }
    if args.json {
        return json_report(&hw, context);
    }
    if args.fast || !std::io::stdout().is_terminal() || !std::io::stdin().is_terminal() {
        return show_report(&hw, context, args.fast);
    }
    if let Some(c) = args.context {
        let mut st = st;
        st.settings.context = c;
        st.save();
    }
    if let Err(e) = tui::run(hw) {
        eprintln!("tokenpati: {e}");
        std::process::exit(1);
    }
}

fn rank(hw: &hardware::Hardware, context: u32) -> Vec<verdict::Ranked> {
    let estimates = budget::estimate_all(hw, catalog::CATALOG, context);
    verdict::rank(verdict::best_quant_per_model(&estimates))
}

fn json_report(hw: &hardware::Hardware, context: u32) {
    let ranked = rank(hw, context);
    let win = verdict::winner(&ranked);
    let report = Report {
        hardware: hw,
        winner: win.map(|r| r.estimate.model.name),
        ranked: ranked
            .iter()
            .map(|r| Row {
                rank: r.rank,
                model: r.estimate.model.name,
                quant: r.estimate.quant.tag(),
                weight_gib: (r.estimate.weight_gib * 100.0).round() / 100.0,
                kv_gib: (r.estimate.kv_gib * 100.0).round() / 100.0,
                fits: r.estimate.fits,
                offload: r.estimate.offload,
                tok_s: (r.estimate.tok_s * 10.0).round() / 10.0,
                max_context: r.estimate.max_context,
                verdict: r.verdict.label(),
            })
            .collect(),
    };
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

fn show_report(hw: &hardware::Hardware, context: u32, fast: bool) {
    let ranked = rank(hw, context);
    show::banner();
    show::wait_for_it(fast);
    show::hot_seat(hw);
    println!();
    show::leaderboard(hw, &ranked);
    println!();
    match verdict::winner(&ranked) {
        Some(r) => show::final_answer(hw, r),
        None => show::nobody_wins(),
    }
    println!();
}

/// Headless launch: same five steps as the app, printed as lines.
fn serve(
    hw: &hardware::Hardware,
    st: &state::State,
    context: u32,
    model: Option<String>,
    engine: Option<String>,
    quant: Option<String>,
    port: Option<u16>,
) -> i32 {
    use engine::EngineKind;
    let ranked = rank(hw, context);
    let pick = match &model {
        Some(q) => {
            let q = q.to_lowercase().replace(['-', '_'], " ");
            ranked.iter().find(|r| {
                r.estimate
                    .model
                    .name
                    .to_lowercase()
                    .replace('-', " ")
                    .contains(&q)
            })
        }
        None => verdict::winner(&ranked),
    };
    let Some(r) = pick else {
        eprintln!(
            "no catalog model matches {:?}. known: {}",
            model.unwrap_or_default(),
            catalog::CATALOG
                .iter()
                .map(|m| m.name)
                .collect::<Vec<_>>()
                .join(", ")
        );
        return 2;
    };
    let q = match quant {
        Some(t) => match budget::Quant::ALL
            .iter()
            .find(|x| x.tag().eq_ignore_ascii_case(&t))
        {
            Some(q) => *q,
            None => {
                eprintln!("unknown quant {t}. use one of Q4_K_M Q5_K_M Q6_K Q8_0 FP16");
                return 2;
            }
        },
        None => r.estimate.quant,
    };
    let m = r.estimate.model;
    let e = budget::estimate(hw, &m, q, context);
    let source = pipeline::Source::Catalog { model: m, quant: q };
    let options = pipeline::LaunchPlan::engines_for(&source);
    let infos = engine::detect_all();
    let kind = match engine.as_deref().and_then(EngineKind::from_name) {
        Some(k) => k,
        None => {
            let usable = |k: EngineKind| infos.iter().any(|i| i.kind == k && i.usable());
            let order = if hw.is_apple() {
                [EngineKind::Mlx, EngineKind::Ollama, EngineKind::LlamaCpp]
            } else {
                [
                    EngineKind::Ollama,
                    EngineKind::LlamaCpp,
                    EngineKind::LlamaCpp,
                ]
            };
            order
                .into_iter()
                .find(|k| usable(*k) && options.contains(k))
                .unwrap_or(EngineKind::LlamaCpp)
        }
    };
    let plan = pipeline::LaunchPlan {
        name: format!("{} {}", m.name, q.tag()),
        engine: kind,
        source,
        context: context.min(m.max_context),
        gpu_layers: e.gpu_layers.filter(|_| hw.has_discrete_gpu()),
        predicted_tok_s: Some(e.tok_s),
        predicted_gib: Some(e.total_gib),
        download_bytes: None,
        measure_key: Some(state::State::key(m.name, q.tag())),
        port: port.unwrap_or(st.settings.port),
        variant: engine::llamacpp::Variant::default_for(hw),
        temperature: st.settings.temperature,
    };
    println!("\x1b[1;38;5;215mtokenpati serve\x1b[0m {} via {} · {} ctx · predicted ~{:.0} tok/s, {:.1} GiB", plan.name, kind.name(), plan.context, e.tok_s, e.total_gib);
    let (tx, rx) = std::sync::mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    {
        let c = cancel.clone();
        let _ = ctrlc_hook(c);
    }
    let emit: Arc<dyn Fn(pipeline::JobEvent) + Send + Sync> = Arc::new(move |ev| {
        let _ = tx.send(ev);
    });
    let c2 = cancel.clone();
    std::thread::spawn(move || pipeline::run(plan, c2, emit));
    let mut session = None;
    let mut last_pct = -1i64;
    let mut ok = false;
    for ev in rx {
        use pipeline::JobEvent::*;
        match ev {
            Start(i, d) => println!(
                "\x1b[1;38;5;215m[{}/5] {}\x1b[0m {d}",
                i + 1,
                pipeline::STEPS[i]
            ),
            Progress(_, p, label) if p.total > 0 => {
                let pct = (p.done * 100 / p.total) as i64;
                if pct != last_pct && pct % 5 == 0 {
                    println!(
                        "      {pct:>3}%  {} / {}  {}",
                        state::human_bytes(p.done),
                        state::human_bytes(p.total),
                        label
                    );
                    last_pct = pct;
                }
            }
            Progress(..) => {}
            Line(_, l) => println!("      \x1b[2m{l}\x1b[0m"),
            Done(_, d) => println!("      \x1b[32m✓ {d}\x1b[0m"),
            Skip(_, d) => println!("      \x1b[34m↷ {d}\x1b[0m"),
            Fail(i, err) => eprintln!(
                "      \x1b[31m✗ {} failed: {err}\x1b[0m",
                pipeline::STEPS[i]
            ),
            Measured { .. } => {}
            Ready(s) => {
                println!(
                    "\n\x1b[1;32m● {} is serving at {}\x1b[0m  (OpenAI-compatible; ctrl-c to stop)",
                    s.name,
                    s.openai_url()
                );
                session = Some(s);
            }
            Finished(o) => {
                ok = o;
                break;
            }
        }
    }
    if !ok {
        return 1;
    }
    // Keep serving until ctrl-c.
    while !cancel.load(std::sync::atomic::Ordering::Relaxed) {
        std::thread::sleep(std::time::Duration::from_millis(200));
        // ctrl-c reaches the server too (same process group); that exit is expected
        if cancel.load(std::sync::atomic::Ordering::Relaxed) {
            break;
        }
        if let Some(s) = session.as_mut() {
            if let Some(Ok(Some(status))) = s.child.as_mut().map(|c| c.try_wait()) {
                eprintln!("server exited: {status}");
                return 1;
            }
        }
    }
    if let Some(mut s) = session {
        println!("\nstopping {}", s.name);
        s.stop();
    }
    0
}

/// SIGINT sets the flag instead of killing us, so the server is shut down cleanly.
fn ctrlc_hook(flag: Arc<AtomicBool>) -> Result<(), ()> {
    #[cfg(unix)]
    {
        static FLAG: std::sync::OnceLock<Arc<AtomicBool>> = std::sync::OnceLock::new();
        let _ = FLAG.set(flag);
        extern "C" fn on_sigint(_: i32) {
            if let Some(f) = FLAG.get() {
                f.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
        extern "C" {
            fn signal(sig: i32, handler: extern "C" fn(i32)) -> usize;
        }
        unsafe {
            signal(2, on_sigint);
        }
    }
    #[cfg(not(unix))]
    let _ = flag;
    Ok(())
}
