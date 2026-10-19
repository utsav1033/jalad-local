//! mlx-lm on Apple silicon: we download the mlx-community weights, then run `mlx_lm.server`.

use super::{EngineInfo, EngineKind};
use crate::net;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// How to launch the server: the console script if it is on PATH, else `python3 -m mlx_lm.server`.
pub fn launcher() -> Option<(PathBuf, Vec<String>)> {
    if let Some(p) = super::which("mlx_lm.server") {
        return Some((p, vec![]));
    }
    let py = super::which("python3")?;
    let ok = Command::new(&py)
        .args(["-c", "import mlx_lm"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    ok.then(|| (py, vec!["-m".into(), "mlx_lm.server".into()]))
}

pub fn detect() -> EngineInfo {
    let supported = EngineKind::Mlx.supported_here();
    let l = if supported { launcher() } else { None };
    EngineInfo {
        kind: EngineKind::Mlx,
        installed: l.is_some(),
        running: false,
        version: None,
        detail: match (&l, supported) {
            (_, false) => "Apple silicon only".into(),
            (Some(_), _) => "mlx-lm ready".into(),
            (None, _) => "not installed. press i to run: pip install -U mlx-lm".into(),
        },
        path: l.map(|x| x.0),
    }
}

/// The shell command that installs mlx-lm, preferring uv or pipx when present.
pub fn install_command() -> String {
    if super::which("uv").is_some() {
        "uv tool install --upgrade mlx-lm".into()
    } else if super::which("pipx").is_some() {
        "pipx install mlx-lm".into()
    } else {
        "python3 -m pip install --user -U mlx-lm".into()
    }
}

pub fn local_dir(repo: &str) -> PathBuf {
    crate::state::models_dir().join("mlx").join(repo)
}

/// Download every file of an mlx-community repo into our models folder.
pub fn download_repo(
    repo: &str,
    cancel: &Arc<AtomicBool>,
    mut log: impl FnMut(String),
    mut progress: impl FnMut(net::Progress),
) -> Result<PathBuf, String> {
    let agent = net::agent_for(&crate::hf::base());
    let files: Vec<_> = crate::hf::tree(repo)?
        .into_iter()
        .filter(|e| e.kind == "file" && !e.path.ends_with(".md") && !e.path.starts_with('.'))
        .collect();
    if files.is_empty() {
        return Err(format!("{repo} has no files"));
    }
    let total: u64 = files.iter().map(|f| f.size).sum();
    let dest = local_dir(repo);
    let mut base = 0u64;
    for f in &files {
        log(format!(
            "{}  ({})",
            f.path,
            crate::state::human_bytes(f.size)
        ));
        net::download(
            &agent,
            &crate::hf::file_url(repo, &f.path),
            &dest.join(&f.path),
            Some(f.size).filter(|s| *s > 0),
            cancel,
            |p| {
                progress(net::Progress {
                    done: base + p.done,
                    total,
                    rate: p.rate,
                })
            },
        )?;
        base += f.size;
    }
    Ok(dest)
}

pub fn serve(
    model_dir: &Path,
    port: u16,
    on_line: impl Fn(String) + Send + Sync + 'static,
) -> Result<Child, String> {
    let (bin, mut args) = launcher().ok_or("mlx-lm is not installed")?;
    args.extend([
        "--model".into(),
        model_dir.display().to_string(),
        "--host".into(),
        "127.0.0.1".into(),
        "--port".into(),
        port.to_string(),
    ]);
    on_line(format!("$ {} {}", bin.display(), args.join(" ")));
    let mut child = Command::new(bin)
        .args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("could not start mlx_lm.server: {e}"))?;
    super::llamacpp::pipe_lines(&mut child, "mlx.log", on_line);
    Ok(child)
}

/// Folders under `root` that look like a model (have a config.json).
pub fn find_models(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    fn walk(d: &Path, depth: usize, out: &mut Vec<PathBuf>) {
        if depth == 0 {
            return;
        }
        if d.join("config.json").is_file() {
            out.push(d.to_path_buf());
            return;
        }
        if let Ok(rd) = std::fs::read_dir(d) {
            for e in rd.flatten() {
                if e.path().is_dir() {
                    walk(&e.path(), depth - 1, out);
                }
            }
        }
    }
    walk(root, 4, &mut out);
    out.sort();
    out
}
