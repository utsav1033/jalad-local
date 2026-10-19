//! Launching a model, as five steps you can watch:
//!
//!   1 engine    is the engine there? install or start it if not
//!   2 weights   already on disk? otherwise download them, resumable
//!   3 load      start the server and wait for the model to land in memory
//!   4 warm-up   first tokens, and a real tok/s measured against the prediction
//!   5 ready     the endpoint, for tokenpati's chat or anything that speaks the OpenAI API
//!
//! The runner reports through a callback, so the TUI draws blocks and `tokenpati serve` prints lines.

use crate::budget::Quant;
use crate::catalog::Model;
use crate::engine::{self, llamacpp, mlx, ollama, Api, ChatOpts, EngineKind, Msg, Session};
use crate::hf::{self, GgufChoice};
use crate::net::{self, Progress};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub const STEPS: [&str; 5] = ["Engine", "Weights", "Load", "Warm-up", "Ready"];

#[derive(Debug, Clone)]
pub enum Source {
    /// A catalog contestant at a chosen quant; the engine decides where the weights come from.
    Catalog {
        model: Model,
        quant: Quant,
    },
    /// A file (or split set) picked from a Hugging Face GGUF repo.
    HfGguf {
        choice: GgufChoice,
    },
    LocalGguf {
        path: PathBuf,
    },
    OllamaTag {
        tag: String,
    },
    LocalMlx {
        path: PathBuf,
    },
}

#[derive(Debug, Clone)]
pub struct LaunchPlan {
    pub name: String,
    pub engine: EngineKind,
    pub source: Source,
    pub context: u32,
    /// llama.cpp -ngl. None lets llama.cpp decide.
    pub gpu_layers: Option<u32>,
    pub predicted_tok_s: Option<f64>,
    pub predicted_gib: Option<f64>,
    /// Download size if not already on disk (best guess, shown before you commit).
    pub download_bytes: Option<u64>,
    pub measure_key: Option<String>,
    pub port: u16,
    /// Prebuilt to fetch if llama.cpp is missing.
    pub variant: llamacpp::Variant,
    pub temperature: f32,
}

impl LaunchPlan {
    /// What the weights will be called in the engine, for the plan screen.
    pub fn artifact(&self) -> String {
        match (&self.source, self.engine) {
            (Source::Catalog { model, quant }, EngineKind::Ollama) => {
                format!("ollama: {}", ollama_tag(model, *quant))
            }
            (Source::Catalog { model, quant }, EngineKind::LlamaCpp) => {
                format!("hf: {}  [{}]", model.gguf, quant.tag())
            }
            (Source::Catalog { model, quant }, EngineKind::Mlx) => {
                format!(
                    "hf: mlx-community/{}-{}",
                    model.mlx,
                    quant.mlx_tag().unwrap_or("4bit")
                )
            }
            (Source::HfGguf { choice }, EngineKind::Ollama) => {
                format!("ollama: {}", hf_ollama_tag(choice))
            }
            (Source::HfGguf { choice }, _) => format!("hf: {}/{}", choice.repo, choice.label()),
            (Source::LocalGguf { path }, _) => path.display().to_string(),
            (Source::OllamaTag { tag }, _) => format!("ollama: {tag}"),
            (Source::LocalMlx { path }, _) => path.display().to_string(),
        }
    }

    /// Engines that can run this source.
    pub fn engines_for(source: &Source) -> Vec<EngineKind> {
        match source {
            Source::Catalog { quant, .. } => {
                let mut v = vec![EngineKind::Ollama, EngineKind::LlamaCpp];
                if EngineKind::Mlx.supported_here() && quant.mlx_tag().is_some() {
                    v.push(EngineKind::Mlx);
                }
                v
            }
            Source::HfGguf { .. } => vec![EngineKind::LlamaCpp, EngineKind::Ollama],
            Source::LocalGguf { .. } => vec![EngineKind::LlamaCpp],
            Source::OllamaTag { .. } => vec![EngineKind::Ollama],
            Source::LocalMlx { .. } => vec![EngineKind::Mlx],
        }
    }
}

pub fn ollama_tag(model: &Model, quant: Quant) -> String {
    format!("{}-{}", model.ollama, quant.ollama_tag())
}

/// ollama pulls straight from Hugging Face with hf.co/<repo>:<quant>.
pub fn hf_ollama_tag(c: &GgufChoice) -> String {
    match &c.quant {
        Some(q) => format!("hf.co/{}:{}", c.repo, q),
        None => format!("hf.co/{}", c.repo),
    }
}

#[derive(Debug)]
pub enum JobEvent {
    Start(usize, String),
    Progress(usize, Progress, String),
    Line(usize, String),
    Done(usize, String),
    Skip(usize, String),
    Fail(usize, String),
    Measured {
        tok_s: f64,
        predicted: Option<f64>,
        key: Option<String>,
        engine: String,
    },
    Ready(Session),
    Finished(bool),
}

struct Steps<'a> {
    emit: &'a (dyn Fn(JobEvent) + Send + Sync),
    cancel: &'a Arc<AtomicBool>,
}

impl Steps<'_> {
    fn start(&self, i: usize, s: impl Into<String>) {
        (self.emit)(JobEvent::Start(i, s.into()));
    }
    fn line(&self, i: usize, s: impl Into<String>) {
        (self.emit)(JobEvent::Line(i, s.into()));
    }
    fn done(&self, i: usize, s: impl Into<String>) {
        (self.emit)(JobEvent::Done(i, s.into()));
    }
    fn skip(&self, i: usize, s: impl Into<String>) {
        (self.emit)(JobEvent::Skip(i, s.into()));
    }
    fn progress(&self, i: usize, p: Progress, label: impl Into<String>) {
        (self.emit)(JobEvent::Progress(i, p, label.into()));
    }
    fn check(&self) -> Result<(), String> {
        if self.cancel.load(Ordering::Relaxed) {
            Err("cancelled".into())
        } else {
            Ok(())
        }
    }
}

/// Run the whole launch. Blocks; call it from a worker thread.
pub fn run(plan: LaunchPlan, cancel: Arc<AtomicBool>, emit: Arc<dyn Fn(JobEvent) + Send + Sync>) {
    let s = Steps {
        emit: emit.as_ref(),
        cancel: &cancel,
    };
    let mut current = 0usize;
    let result = launch(&plan, &s, &mut current, emit.clone());
    match result {
        Ok(()) => emit(JobEvent::Finished(true)),
        Err(e) => {
            emit(JobEvent::Fail(current, e));
            emit(JobEvent::Finished(false));
        }
    }
}

/// What the weights step hands to the load step.
enum Weights {
    OllamaTag(String),
    Gguf(PathBuf),
    MlxDir(PathBuf),
}

fn launch(
    plan: &LaunchPlan,
    s: &Steps,
    cur: &mut usize,
    emit: Arc<dyn Fn(JobEvent) + Send + Sync>,
) -> Result<(), String> {
    // ------------------------------------------------ 1 engine
    *cur = 0;
    s.start(0, format!("checking {}", plan.engine.name()));
    let llama_bin: Option<PathBuf> = match plan.engine {
        EngineKind::Ollama => {
            let v = ollama::ensure_server(|l| s.line(0, l))?;
            s.done(0, format!("ollama {v} at {}", ollama::host()));
            None
        }
        EngineKind::LlamaCpp => {
            let bin = match llamacpp::server_binary() {
                Some(b) => b,
                None => {
                    s.line(
                        0,
                        format!(
                            "llama-server not found. installing the official {} build",
                            plan.variant.name()
                        ),
                    );
                    llamacpp::install(
                        plan.variant,
                        s.cancel,
                        |l| s.line(0, l),
                        |p| s.progress(0, p, "llama.cpp"),
                    )?
                }
            };
            let v = llamacpp::version_of(&bin).unwrap_or_else(|| "?".into());
            s.done(0, format!("llama-server {v}  ({})", bin.display()));
            Some(bin)
        }
        EngineKind::Mlx => {
            let (bin, _) = mlx::launcher().ok_or_else(|| {
                format!("mlx-lm is not installed. run: {}", mlx::install_command())
            })?;
            s.done(0, format!("mlx-lm ({})", bin.display()));
            None
        }
    };
    s.check()?;

    // ------------------------------------------------ 2 weights
    *cur = 1;
    s.start(1, "looking for the weights");
    let weights = fetch_weights(plan, s)?;
    s.check()?;

    // ------------------------------------------------ 3 load
    *cur = 2;
    s.start(
        2,
        format!("loading with {} tokens of context", plan.context),
    );
    let t0 = Instant::now();
    let session = match (&weights, plan.engine) {
        (Weights::OllamaTag(tag), EngineKind::Ollama) => {
            let base = ollama::host();
            let (tx, rx) = std::sync::mpsc::channel();
            let (b2, t2, ctx) = (base.clone(), tag.clone(), plan.context);
            std::thread::spawn(move || {
                let _ = tx.send(ollama::load(&b2, &t2, ctx));
            });
            let load = loop {
                match rx.recv_timeout(Duration::from_millis(500)) {
                    Ok(r) => break r,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        s.check()?;
                        s.progress(
                            2,
                            Progress {
                                done: 0,
                                total: 0,
                                rate: 0.0,
                            },
                            format!(
                                "reading weights into memory  {:.0}s",
                                t0.elapsed().as_secs_f64()
                            ),
                        );
                    }
                    Err(_) => break Err("ollama load thread died".into()),
                }
            }?;
            if let Ok(ps) = ollama::ps(&base) {
                if let Some(m) = ps
                    .iter()
                    .find(|m| &m.name == tag || m.name == format!("{tag}:latest"))
                {
                    let gpu = if m.size > 0 {
                        m.size_vram as f64 / m.size as f64 * 100.0
                    } else {
                        0.0
                    };
                    s.line(
                        2,
                        format!(
                            "resident: {}  ({gpu:.0}% on GPU)",
                            crate::state::human_bytes(m.size)
                        ),
                    );
                }
            }
            s.done(
                2,
                format!(
                    "loaded in {:.1}s",
                    load.as_secs_f64().max(t0.elapsed().as_secs_f64())
                ),
            );
            Session {
                name: plan.name.clone(),
                engine: EngineKind::Ollama,
                api: Api::Ollama,
                base_url: base,
                model_id: tag.clone(),
                context: plan.context,
                child: None,
                predicted_tok_s: plan.predicted_tok_s,
                measure_key: plan.measure_key.clone(),
                started: Instant::now(),
            }
        }
        (Weights::Gguf(path), EngineKind::LlamaCpp) => {
            let bin = llama_bin.clone().ok_or("no llama-server")?;
            let port = engine::free_port(plan.port);
            let e2 = emit.clone();
            let child = llamacpp::serve(
                &bin,
                &llamacpp::ServeOpts {
                    model: path,
                    context: plan.context,
                    gpu_layers: plan.gpu_layers,
                    port,
                    threads: None,
                },
                move |l| e2(JobEvent::Line(2, l)),
            )?;
            let base = format!("http://127.0.0.1:{port}");
            let mut session = Session {
                name: plan.name.clone(),
                engine: EngineKind::LlamaCpp,
                api: Api::OpenAi,
                base_url: base.clone(),
                model_id: path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("model")
                    .to_string(),
                context: plan.context,
                child: Some(child),
                predicted_tok_s: plan.predicted_tok_s,
                measure_key: plan.measure_key.clone(),
                started: Instant::now(),
            };
            wait_server(&mut session, "/health", s, t0)?;
            session
        }
        (Weights::MlxDir(dir), EngineKind::Mlx) => {
            let port = engine::free_port(plan.port);
            let e2 = emit.clone();
            let child = mlx::serve(dir, port, move |l| e2(JobEvent::Line(2, l)))?;
            let mut session = Session {
                name: plan.name.clone(),
                engine: EngineKind::Mlx,
                api: Api::OpenAi,
                base_url: format!("http://127.0.0.1:{port}"),
                model_id: dir.display().to_string(),
                context: plan.context,
                child: Some(child),
                predicted_tok_s: plan.predicted_tok_s,
                measure_key: plan.measure_key.clone(),
                started: Instant::now(),
            };
            wait_server(&mut session, "/v1/models", s, t0)?;
            session
        }
        _ => return Err("these weights do not match this engine".into()),
    };
    s.check()?;

    // ------------------------------------------------ 4 warm-up
    *cur = 3;
    s.start(3, "first tokens, measuring real speed");
    let msgs = vec![Msg::new(
        "user",
        "Introduce yourself in one short sentence.",
    )];
    let opts = ChatOpts {
        temperature: plan.temperature,
        num_ctx: plan.context,
        max_tokens: Some(64),
    };
    let mut said = String::new();
    let stats = engine::chat(
        &session.api,
        &session.base_url,
        &session.model_id,
        &msgs,
        &opts,
        s.cancel,
        |c| {
            if let engine::Chunk::Content(t) = c {
                said.push_str(&t);
            }
        },
    )
    .map_err(|e| format!("first reply failed: {e}"))?;
    let said = said.split_whitespace().collect::<Vec<_>>().join(" ");
    if !said.is_empty() {
        s.line(
            3,
            format!(
                "\u{201c}{}\u{201d}",
                said.chars().take(160).collect::<String>()
            ),
        );
    }
    let vs = match plan.predicted_tok_s {
        Some(p) if p > 0.0 => format!(
            "  (predicted ~{p:.0}, {:+.0}%)",
            (stats.tok_s / p - 1.0) * 100.0
        ),
        _ => String::new(),
    };
    s.done(
        3,
        format!(
            "{:.1} tok/s, first token in {} ms{vs}",
            stats.tok_s, stats.ttft_ms
        ),
    );
    if stats.tok_s > 0.0 && stats.tokens >= 8 {
        emit(JobEvent::Measured {
            tok_s: stats.tok_s,
            predicted: plan.predicted_tok_s,
            key: plan.measure_key.clone(),
            engine: plan.engine.name().into(),
        });
    }

    // ------------------------------------------------ 5 ready
    *cur = 4;
    s.start(4, "");
    s.line(
        4,
        format!("OpenAI-compatible endpoint: {}", session.openai_url()),
    );
    s.done(4, format!("{} is live", plan.name));
    emit(JobEvent::Ready(session));
    Ok(())
}

fn wait_server(session: &mut Session, path: &str, s: &Steps, t0: Instant) -> Result<(), String> {
    let base = session.base_url.clone();
    let child = session.child.as_mut();
    let mut child = child;
    let took = engine::openai::wait_ready(
        &base,
        path,
        Duration::from_secs(1800),
        s.cancel,
        || match child.as_mut().map(|c| c.try_wait()) {
            Some(Ok(Some(st))) => Err(format!(
                "the server exited ({st}). the last lines above say why"
            )),
            _ => Ok(()),
        },
        |el| {
            s.progress(
                2,
                Progress {
                    done: 0,
                    total: 0,
                    rate: 0.0,
                },
                format!("reading weights into memory  {:.0}s", el.as_secs_f64()),
            )
        },
    )?;
    if let Some(pid) = session.pid() {
        if let Some(rss) = crate::hardware::process_rss_gib(pid) {
            let gpu = if crate::engine::which("nvidia-smi").is_some() {
                " (GPU memory is extra)"
            } else {
                ""
            };
            s.line(
                2,
                format!(
                    "server process uses {} of RAM{gpu}",
                    crate::state::human_bytes((rss * (1u64 << 30) as f64) as u64)
                ),
            );
        }
    }
    s.done(
        2,
        format!(
            "loaded in {:.1}s, serving at {base}",
            took.max(t0.elapsed()).as_secs_f64()
        ),
    );
    Ok(())
}

fn fetch_weights(plan: &LaunchPlan, s: &Steps) -> Result<Weights, String> {
    match (&plan.source, plan.engine) {
        (Source::OllamaTag { tag }, _) => {
            s.skip(1, format!("{tag} is already in ollama"));
            Ok(Weights::OllamaTag(tag.clone()))
        }
        (Source::LocalGguf { path }, _) => {
            s.skip(1, format!("on disk: {}", path.display()));
            Ok(Weights::Gguf(path.clone()))
        }
        (Source::LocalMlx { path }, _) => {
            s.skip(1, format!("on disk: {}", path.display()));
            Ok(Weights::MlxDir(path.clone()))
        }
        (Source::Catalog { model, quant }, EngineKind::Ollama) => {
            let base = ollama::host();
            // Not every quant exists in ollama's library; fall back gracefully and say so.
            let candidates = [
                ollama_tag(model, *quant),
                format!(
                    "hf.co/{}:{}",
                    model.gguf,
                    quant.tag().replace("FP16", "F16")
                ),
                model.ollama.to_string(),
            ];
            for (n, tag) in candidates.iter().enumerate() {
                if ollama::has(&base, tag) {
                    s.skip(1, format!("{tag} is already downloaded"));
                    return Ok(Weights::OllamaTag(tag.clone()));
                }
                if n > 0 {
                    s.line(1, format!("trying {tag}"));
                }
                match pull(&base, tag, s) {
                    Ok(()) => {
                        if n == 2 {
                            s.line(1, "note: this is ollama's default quant for the model, not the one you picked");
                        }
                        s.done(1, format!("pulled {tag}"));
                        return Ok(Weights::OllamaTag(tag.clone()));
                    }
                    Err(e) if n < 2 && !s.cancel.load(Ordering::Relaxed) && looks_missing(&e) => {
                        s.line(1, format!("{tag}: {e}"));
                    }
                    Err(e) => return Err(e),
                }
            }
            Err("ollama could not find any build of this model".into())
        }
        (Source::HfGguf { choice }, EngineKind::Ollama) => {
            let base = ollama::host();
            let tag = hf_ollama_tag(choice);
            if ollama::has(&base, &tag) {
                s.skip(1, format!("{tag} is already downloaded"));
            } else {
                pull(&base, &tag, s)?;
                s.done(1, format!("pulled {tag}"));
            }
            Ok(Weights::OllamaTag(tag))
        }
        (Source::Catalog { model, quant }, EngineKind::LlamaCpp) => {
            s.line(1, format!("listing files in {}", model.gguf));
            let entries = hf::tree(model.gguf)?;
            let choices = hf::gguf_choices(model.gguf, &entries);
            let choice = hf::pick(&choices, quant.tag()).ok_or_else(|| {
                let have: Vec<String> = choices.iter().filter_map(|c| c.quant.clone()).collect();
                format!(
                    "{} has no {} file. it has: {}",
                    model.gguf,
                    quant.tag(),
                    have.join(", ")
                )
            })?;
            download_gguf(&choice, s)
        }
        (Source::HfGguf { choice }, EngineKind::LlamaCpp) => download_gguf(choice, s),
        (Source::Catalog { model, quant }, EngineKind::Mlx) => {
            let tag = quant.mlx_tag().ok_or("mlx has no build at this quant")?;
            let repo = format!("mlx-community/{}-{}", model.mlx, tag);
            let dir = mlx::local_dir(&repo);
            if dir.join("config.json").is_file()
                && !dir
                    .read_dir()
                    .map(|mut d| {
                        d.any(|e| {
                            e.map(|e| e.path().extension().is_some_and(|x| x == "part"))
                                .unwrap_or(false)
                        })
                    })
                    .unwrap_or(false)
            {
                s.skip(1, format!("on disk: {}", dir.display()));
                return Ok(Weights::MlxDir(dir));
            }
            let d = mlx::download_repo(
                &repo,
                s.cancel,
                |l| s.line(1, l),
                |p| s.progress(1, p, repo.clone()),
            )?;
            s.done(1, format!("{repo} downloaded"));
            Ok(Weights::MlxDir(d))
        }
        _ => Err("this engine cannot use these weights".into()),
    }
}

fn looks_missing(e: &str) -> bool {
    let e = e.to_lowercase();
    e.contains("not found")
        || e.contains("does not exist")
        || e.contains("404")
        || e.contains("manifest unknown")
}

fn pull(base: &str, tag: &str, s: &Steps) -> Result<(), String> {
    s.line(1, format!("ollama pull {tag}"));
    let started = Instant::now();
    let mut last_status = String::new();
    ollama::pull(base, tag, s.cancel, |p| {
        if p.status != last_status && !p.status.starts_with("pulling ") {
            s.line(1, p.status.clone());
        }
        last_status = p.status.clone();
        let secs = started.elapsed().as_secs_f64().max(0.1);
        s.progress(
            1,
            Progress {
                done: p.done,
                total: p.total,
                rate: p.done as f64 / secs,
            },
            tag.to_string(),
        );
    })
}

pub fn gguf_dest(repo: &str, path: &str) -> PathBuf {
    crate::state::models_dir()
        .join("gguf")
        .join(repo)
        .join(path)
}

fn download_gguf(choice: &GgufChoice, s: &Steps) -> Result<Weights, String> {
    let first = gguf_dest(&choice.repo, &choice.files[0].0);
    let all_there = choice.files.iter().all(|(p, size)| {
        std::fs::metadata(gguf_dest(&choice.repo, p))
            .map(|m| *size == 0 || m.len() == *size)
            .unwrap_or(false)
    });
    if all_there {
        s.skip(1, format!("on disk: {}", first.display()));
        return Ok(Weights::Gguf(first));
    }
    let free = crate::state::free_disk_bytes(&crate::state::models_dir());
    if let Some(free) = free {
        if free < choice.size() {
            return Err(format!(
                "not enough disk: need {}, {} free at {}",
                crate::state::human_bytes(choice.size()),
                crate::state::human_bytes(free),
                crate::state::models_dir().display()
            ));
        }
    }
    let agent = net::agent_for(&choice.url(0));
    let total = choice.size();
    let mut base = 0u64;
    for (i, (path, size)) in choice.files.iter().enumerate() {
        s.line(1, format!("{}  {}", path, crate::state::human_bytes(*size)));
        let dest = gguf_dest(&choice.repo, path);
        net::download(
            &agent,
            &choice.url(i),
            &dest,
            Some(*size).filter(|x| *x > 0),
            s.cancel,
            |p| {
                s.progress(
                    1,
                    Progress {
                        done: base + p.done,
                        total,
                        rate: p.rate,
                    },
                    choice.label(),
                )
            },
        )?;
        base += size;
    }
    s.done(
        1,
        format!(
            "{} saved to {}",
            crate::state::human_bytes(total),
            first
                .parent()
                .map(|p| p.display().to_string())
                .unwrap_or_default()
        ),
    );
    Ok(Weights::Gguf(first))
}
