//! Engines are the programs that actually run a model. tokenpati drives three:
//!
//!   ollama     easiest. one install, pulls its own weights, runs everywhere.
//!   llama.cpp  most control. tokenpati downloads the binary and the GGUF itself; partial GPU offload.
//!   mlx        fastest on Apple silicon.
//!
//! Each one ends up as an HTTP server on localhost, so once a model is loaded anything that speaks
//! the OpenAI API can use it, not just tokenpati.

pub mod llamacpp;
pub mod mlx;
pub mod ollama;
pub mod openai;

use std::path::{Path, PathBuf};
use std::process::Child;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EngineKind {
    Ollama,
    LlamaCpp,
    Mlx,
}

impl EngineKind {
    pub const ALL: [EngineKind; 3] = [EngineKind::Ollama, EngineKind::LlamaCpp, EngineKind::Mlx];

    pub fn name(self) -> &'static str {
        match self {
            EngineKind::Ollama => "ollama",
            EngineKind::LlamaCpp => "llama.cpp",
            EngineKind::Mlx => "mlx",
        }
    }

    pub fn from_name(s: &str) -> Option<EngineKind> {
        EngineKind::ALL.into_iter().find(|k| k.name() == s)
    }

    pub fn blurb(self) -> &'static str {
        match self {
            EngineKind::Ollama => {
                "easiest. pulls its own weights, keeps models warm, runs everywhere."
            }
            EngineKind::LlamaCpp => {
                "most control. any GGUF on Hugging Face, exact quant, GPU layer split."
            }
            EngineKind::Mlx => {
                "fastest on Apple silicon. runs mlx-community weights natively on the GPU."
            }
        }
    }

    /// Only mlx is platform bound.
    pub fn supported_here(self) -> bool {
        self != EngineKind::Mlx || cfg!(all(target_os = "macos", target_arch = "aarch64"))
    }
}

#[derive(Debug, Clone)]
pub struct EngineInfo {
    pub kind: EngineKind,
    pub installed: bool,
    pub path: Option<PathBuf>,
    pub version: Option<String>,
    /// Server up right now (ollama) or a model is being served by us (llama.cpp / mlx).
    pub running: bool,
    pub detail: String,
}

impl EngineInfo {
    pub fn usable(&self) -> bool {
        self.installed && self.kind.supported_here()
    }
}

pub fn detect_all() -> Vec<EngineInfo> {
    vec![ollama::detect(), llamacpp::detect(), mlx::detect()]
}

/// Which API a running model speaks.
#[derive(Debug, Clone, PartialEq)]
pub enum Api {
    /// ollama's native /api/chat: exact token timings, thinking field.
    Ollama,
    /// /v1/chat/completions with SSE.
    OpenAi,
}

/// A model that is loaded and answering.
#[derive(Debug)]
pub struct Session {
    pub name: String,
    pub engine: EngineKind,
    pub api: Api,
    /// e.g. http://127.0.0.1:11434
    pub base_url: String,
    /// What to put in the request's "model" field.
    pub model_id: String,
    pub context: u32,
    /// The server we spawned, if we spawned one (llama.cpp, mlx). ollama owns its own.
    pub child: Option<Child>,
    pub predicted_tok_s: Option<f64>,
    /// Key into State::measurements.
    pub measure_key: Option<String>,
    pub started: std::time::Instant,
}

impl Session {
    pub fn openai_url(&self) -> String {
        format!("{}/v1", self.base_url)
    }

    pub fn pid(&self) -> Option<u32> {
        self.child.as_ref().map(|c| c.id())
    }

    /// Unload the model / stop our server.
    pub fn stop(&mut self) {
        if let Some(mut c) = self.child.take() {
            terminate(&mut c);
        }
        if self.engine == EngineKind::Ollama {
            let _ = ollama::unload(&self.base_url, &self.model_id);
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        if let Some(mut c) = self.child.take() {
            terminate(&mut c);
        }
    }
}

/// SIGTERM, a short grace period, then SIGKILL.
pub fn terminate(c: &mut Child) {
    #[cfg(unix)]
    {
        if let Ok(Some(_)) = c.try_wait() {
            return;
        }
        let _ = std::process::Command::new("kill")
            .arg("-TERM")
            .arg(c.id().to_string())
            .stderr(std::process::Stdio::null())
            .status();
        for _ in 0..30 {
            if let Ok(Some(_)) = c.try_wait() {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }
    let _ = c.kill();
    let _ = c.wait();
}

/// `which`, without the dependency.
pub fn which(bin: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let p = dir.join(bin);
        if is_executable(&p) {
            return Some(p);
        }
    }
    None
}

pub fn is_executable(p: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        p.is_file()
            && p.metadata()
                .map(|m| m.permissions().mode() & 0o111 != 0)
                .unwrap_or(false)
    }
    #[cfg(not(unix))]
    {
        p.is_file()
    }
}

/// First free localhost port at or after `start`.
pub fn free_port(start: u16) -> u16 {
    (start..start.saturating_add(200))
        .find(|p| std::net::TcpListener::bind(("127.0.0.1", *p)).is_ok())
        .unwrap_or(start)
}

/// Where a downloaded model lives, whichever engine owns it.
#[derive(Debug, Clone)]
pub struct LocalModel {
    pub name: String,
    pub engine: EngineKind,
    /// ollama tag, or path to the .gguf / mlx folder.
    pub id: String,
    pub size: u64,
    pub detail: String,
    /// Can tokenpati delete it? (not for LM Studio's folder)
    pub ours: bool,
    pub loaded: bool,
}

/// Everything already on disk: ollama's store, our models folder, and LM Studio's models folder.
pub fn list_local() -> Vec<LocalModel> {
    let mut out = Vec::new();
    let loaded = ollama::ps(&ollama::host()).unwrap_or_default();
    if let Ok(tags) = ollama::tags(&ollama::host()) {
        for t in tags {
            out.push(LocalModel {
                loaded: loaded.iter().any(|p| p.name == t.name),
                detail: format!(
                    "{} {}",
                    t.details.parameter_size, t.details.quantization_level
                )
                .trim()
                .to_string(),
                name: t.name.clone(),
                engine: EngineKind::Ollama,
                id: t.name,
                size: t.size,
                ours: true,
            });
        }
    }
    let gguf_root = crate::state::models_dir().join("gguf");
    for p in llamacpp::find_ggufs(&gguf_root) {
        out.push(gguf_entry(&gguf_root, &p, true));
    }
    for root in lmstudio_dirs() {
        for p in llamacpp::find_ggufs(&root) {
            let mut e = gguf_entry(&root, &p, false);
            e.detail = format!("{}  · from LM Studio", e.detail);
            out.push(e);
        }
    }
    let mlx_root = crate::state::models_dir().join("mlx");
    for p in mlx::find_models(&mlx_root) {
        let rel = p
            .strip_prefix(&mlx_root)
            .unwrap_or(&p)
            .display()
            .to_string();
        out.push(LocalModel {
            name: rel,
            engine: EngineKind::Mlx,
            size: crate::state::dir_size(&p),
            id: p.display().to_string(),
            detail: "mlx".into(),
            ours: true,
            loaded: false,
        });
    }
    out
}

fn gguf_entry(root: &Path, p: &Path, ours: bool) -> LocalModel {
    let file = p
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("?")
        .to_string();
    let rel = p.strip_prefix(root).unwrap_or(p);
    // A split model's size is all its parts.
    let size = llamacpp::split_siblings(p)
        .iter()
        .map(|s| std::fs::metadata(s).map(|m| m.len()).unwrap_or(0))
        .sum();
    let repo = rel
        .parent()
        .map(|r| r.display().to_string())
        .unwrap_or_default();
    LocalModel {
        name: file.trim_end_matches(".gguf").to_string(),
        engine: EngineKind::LlamaCpp,
        id: p.display().to_string(),
        size,
        detail: format!(
            "{}{}",
            crate::budget::quant_from_filename(&file).unwrap_or_else(|| "gguf".into()),
            if repo.is_empty() {
                String::new()
            } else {
                format!("  · {repo}")
            }
        ),
        ours,
        loaded: false,
    }
}

pub fn lmstudio_dirs() -> Vec<PathBuf> {
    let Some(h) = dirs::home_dir() else {
        return vec![];
    };
    [
        h.join(".lmstudio").join("models"),
        h.join(".cache").join("lm-studio").join("models"),
    ]
    .into_iter()
    .filter(|p| p.is_dir())
    .collect()
}

/// Remove a model from disk (or from ollama).
pub fn delete(m: &LocalModel) -> Result<(), String> {
    if !m.ours {
        return Err("this file belongs to LM Studio; delete it there".into());
    }
    match m.engine {
        EngineKind::Ollama => ollama::delete(&ollama::host(), &m.id),
        EngineKind::LlamaCpp => {
            for p in llamacpp::split_siblings(Path::new(&m.id)) {
                std::fs::remove_file(&p).map_err(|e| format!("{}: {e}", p.display()))?;
            }
            Ok(())
        }
        EngineKind::Mlx => std::fs::remove_dir_all(&m.id).map_err(|e| e.to_string()),
    }
}

// ---------------------------------------------------------------- chat, shared by every engine

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct Msg {
    pub role: String,
    pub content: String,
}

impl Msg {
    pub fn new(role: &str, content: impl Into<String>) -> Msg {
        Msg {
            role: role.into(),
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Chunk {
    Content(String),
    Thinking(String),
}

#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub tokens: u64,
    pub tok_s: f64,
    pub ttft_ms: u64,
    pub prompt_tokens: Option<u64>,
    pub prompt_tok_s: Option<f64>,
    /// Timings came from the engine itself rather than our stopwatch.
    pub exact: bool,
}

#[derive(Debug, Clone)]
pub struct ChatOpts {
    pub temperature: f32,
    pub num_ctx: u32,
    pub max_tokens: Option<u32>,
}

/// Stream one reply. `on_chunk` sees text as it arrives; returns timings when the reply ends.
pub fn chat(
    api: &Api,
    base_url: &str,
    model: &str,
    msgs: &[Msg],
    opts: &ChatOpts,
    cancel: &std::sync::Arc<std::sync::atomic::AtomicBool>,
    on_chunk: impl FnMut(Chunk),
) -> Result<Stats, String> {
    match api {
        Api::Ollama => ollama::chat(base_url, model, msgs, opts, cancel, on_chunk),
        Api::OpenAi => openai::chat(base_url, model, msgs, opts, cancel, on_chunk),
    }
}

/// Splits `<think>...</think>` out of a content stream, even when a tag is cut across chunks.
#[derive(Default)]
pub struct ThinkSplitter {
    in_think: bool,
    buf: String,
}

impl ThinkSplitter {
    pub fn push(&mut self, text: &str, mut emit: impl FnMut(Chunk)) {
        self.buf.push_str(text);
        loop {
            let tag = if self.in_think { "</think>" } else { "<think>" };
            if let Some(i) = self.buf.find(tag) {
                let before: String = self.buf[..i].to_string();
                self.out(&before, &mut emit);
                self.buf = self.buf[i + tag.len()..].to_string();
                self.in_think = !self.in_think;
                continue;
            }
            // Hold back anything that could be the start of a tag.
            let keep = (1..tag.len())
                .rev()
                .find(|n| self.buf.ends_with(&tag[..*n]))
                .unwrap_or(0);
            let cut = self.buf.len() - keep;
            let ready: String = self.buf[..cut].to_string();
            self.out(&ready, &mut emit);
            self.buf = self.buf[cut..].to_string();
            break;
        }
    }

    pub fn finish(&mut self, mut emit: impl FnMut(Chunk)) {
        let rest = std::mem::take(&mut self.buf);
        self.out(&rest, &mut emit);
    }

    fn out(&self, s: &str, emit: &mut impl FnMut(Chunk)) {
        if s.is_empty() {
            return;
        }
        emit(if self.in_think {
            Chunk::Thinking(s.to_string())
        } else {
            Chunk::Content(s.to_string())
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn think_tags_split_across_chunks() {
        let mut s = ThinkSplitter::default();
        let mut got = Vec::new();
        for piece in [
            "<thi",
            "nk>hmm, ",
            "let me see</th",
            "ink>\nThe answer",
            " is 4.",
        ] {
            s.push(piece, |c| got.push(c));
        }
        s.finish(|c| got.push(c));
        let thinking: String = got
            .iter()
            .filter_map(|c| {
                if let Chunk::Thinking(t) = c {
                    Some(t.as_str())
                } else {
                    None
                }
            })
            .collect();
        let content: String = got
            .iter()
            .filter_map(|c| {
                if let Chunk::Content(t) = c {
                    Some(t.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(thinking, "hmm, let me see");
        assert_eq!(content, "\nThe answer is 4.");
    }

    #[test]
    fn plain_text_passes_through() {
        let mut s = ThinkSplitter::default();
        let mut got = String::new();
        s.push("a < b and c <", |c| {
            if let Chunk::Content(t) = c {
                got.push_str(&t)
            }
        });
        s.push(" d", |c| {
            if let Chunk::Content(t) = c {
                got.push_str(&t)
            }
        });
        s.finish(|c| {
            if let Chunk::Content(t) = c {
                got.push_str(&t)
            }
        });
        assert_eq!(got, "a < b and c < d");
    }
}
