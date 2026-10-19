//! HTTP: one agent that honours HTTPS_PROXY and the system trust store,
//! plus a resumable, cancellable downloader that reports progress.

use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// An agent for talking to `url`: through HTTPS_PROXY unless NO_PROXY (or loopback) says otherwise.
/// ureq's own env handling ignores NO_PROXY, which breaks local mirrors and corporate setups.
pub fn agent_for(url: &str) -> ureq::Agent {
    let b = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(10))
        .timeout_read(Duration::from_secs(60))
        .user_agent(concat!("tokenpati/", env!("CARGO_PKG_VERSION")));
    let host = host_of(url);
    if bypass_proxy(
        &host,
        &std::env::var("NO_PROXY")
            .or_else(|_| std::env::var("no_proxy"))
            .unwrap_or_default(),
    ) {
        return b.build();
    }
    let proxy = ["HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"]
        .iter()
        .find_map(|k| std::env::var(k).ok().filter(|v| !v.is_empty()))
        .and_then(|p| ureq::Proxy::new(p).ok());
    match proxy {
        Some(p) => b.proxy(p).build(),
        None => b.build(),
    }
}

pub fn host_of(url: &str) -> String {
    let rest = url.split("://").nth(1).unwrap_or(url);
    let hostport = rest.split(['/', '?']).next().unwrap_or("");
    let hostport = hostport.rsplit('@').next().unwrap_or(hostport);
    if let Some(v6) = hostport.strip_prefix('[') {
        return v6.split(']').next().unwrap_or("").to_lowercase();
    }
    hostport.split(':').next().unwrap_or("").to_lowercase()
}

pub fn bypass_proxy(host: &str, no_proxy: &str) -> bool {
    if host == "localhost" || host.starts_with("127.") || host == "::1" {
        return true;
    }
    no_proxy
        .split(',')
        .map(|e| e.trim().to_lowercase())
        .filter(|e| !e.is_empty())
        .any(|e| {
            if e == "*" {
                return true;
            }
            let e = e.trim_start_matches("*.").trim_start_matches('.');
            let e = e.split(':').next().unwrap_or(e);
            host == e || host.ends_with(&format!(".{e}"))
        })
}

/// Local servers never go through a proxy. Reads may wait minutes: a long prompt on a CPU is slow
/// to produce its first token, and loading a 40 GB model is slower still.
pub fn local_agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(3))
        .timeout_read(Duration::from_secs(1800))
        .build()
}

/// A request to Hugging Face, with HF_TOKEN attached when set (gated repos).
pub fn hf_get(agent: &ureq::Agent, url: &str) -> ureq::Request {
    let req = agent.get(url);
    match std::env::var("HF_TOKEN").or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN")) {
        Ok(t) if !t.is_empty() && url.starts_with(&crate::hf::base()) => {
            req.set("Authorization", &format!("Bearer {t}"))
        }
        _ => req,
    }
}

pub fn err_text(e: ureq::Error) -> String {
    match e {
        ureq::Error::Status(code, resp) => {
            let url = resp.get_url().to_string();
            let body = resp.into_string().unwrap_or_default();
            let body: String = body.chars().take(200).collect();
            match code {
                401 | 403 if url.contains("huggingface.co") => {
                    format!("HTTP {code}: this repo is gated or private. Accept its licence on huggingface.co and export HF_TOKEN.")
                }
                404 => format!("HTTP 404: not found ({url})"),
                _ => format!("HTTP {code}: {}", body.trim()),
            }
        }
        ureq::Error::Transport(t) => format!("network: {t}"),
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Progress {
    pub done: u64,
    pub total: u64,
    /// Bytes per second, smoothed.
    pub rate: f64,
}

impl Progress {
    pub fn eta_secs(&self) -> Option<u64> {
        if self.rate <= 1.0 || self.total == 0 {
            return None;
        }
        Some(((self.total.saturating_sub(self.done)) as f64 / self.rate) as u64)
    }
}

pub fn human_secs(s: u64) -> String {
    if s >= 3600 {
        format!("{}h{:02}m", s / 3600, (s % 3600) / 60)
    } else if s >= 60 {
        format!("{}m{:02}s", s / 60, s % 60)
    } else {
        format!("{s}s")
    }
}

/// Download `url` to `dest`, resuming from `dest.part` if a previous attempt was cut off.
/// `on_progress` is called at most ~10 times a second. Returns bytes written in total.
pub fn download(
    agent: &ureq::Agent,
    url: &str,
    dest: &Path,
    expected: Option<u64>,
    cancel: &Arc<AtomicBool>,
    mut on_progress: impl FnMut(Progress),
) -> Result<u64, String> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    if let (Ok(meta), Some(n)) = (std::fs::metadata(dest), expected) {
        if meta.len() == n {
            on_progress(Progress {
                done: n,
                total: n,
                rate: 0.0,
            });
            return Ok(n);
        }
    }
    let part = dest.with_file_name(format!(
        "{}.part",
        dest.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("download")
    ));
    let mut have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    if let Some(n) = expected {
        if have > n {
            have = 0;
        }
    }

    let mut req = hf_get(agent, url);
    if have > 0 {
        req = req.set("Range", &format!("bytes={have}-"));
    }
    let resp = req.call().map_err(err_text)?;
    let resumed = resp.status() == 206;
    if !resumed {
        have = 0;
    }
    let len: Option<u64> = resp.header("Content-Length").and_then(|v| v.parse().ok());
    let total = expected.or(len.map(|l| l + have)).unwrap_or(0);

    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(resumed)
        .truncate(!resumed)
        .open(&part)
        .map_err(|e| format!("open {}: {e}", part.display()))?;

    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 256 * 1024];
    let mut done = have;
    let mut last_emit = Instant::now() - Duration::from_secs(1);
    let mut window_start = Instant::now();
    let mut window_bytes = 0u64;
    let mut rate = 0.0f64;
    loop {
        if cancel.load(Ordering::Relaxed) {
            let _ = file.flush();
            return Err("cancelled. the partial file is kept, so the next attempt resumes".into());
        }
        let n = reader
            .read(&mut buf)
            .map_err(|e| format!("download interrupted: {e}. run it again to resume"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .map_err(|e| format!("write: {e} (disk full?)"))?;
        done += n as u64;
        window_bytes += n as u64;
        let el = window_start.elapsed().as_secs_f64();
        if el >= 0.5 {
            let inst = window_bytes as f64 / el;
            rate = if rate == 0.0 {
                inst
            } else {
                rate * 0.7 + inst * 0.3
            };
            window_start = Instant::now();
            window_bytes = 0;
        }
        if last_emit.elapsed() >= Duration::from_millis(100) {
            on_progress(Progress { done, total, rate });
            last_emit = Instant::now();
        }
    }
    file.flush().map_err(|e| e.to_string())?;
    drop(file);
    if total > 0 && done < total {
        return Err(format!(
            "download stopped at {done} of {total} bytes. run it again to resume"
        ));
    }
    std::fs::rename(&part, dest).map_err(|e| format!("rename: {e}"))?;
    on_progress(Progress {
        done,
        total: total.max(done),
        rate,
    });
    Ok(done)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_proxy_rules() {
        assert_eq!(
            host_of("https://huggingface.co/api/models?x=1"),
            "huggingface.co"
        );
        assert_eq!(host_of("http://user@127.0.0.1:7860/a"), "127.0.0.1");
        assert!(bypass_proxy("127.0.0.1", ""));
        assert!(bypass_proxy("cdn.hf.co", "localhost, .hf.co"));
        assert!(bypass_proxy("hf.co", "*.hf.co"));
        assert!(!bypass_proxy("huggingface.co", "hf.co,github.com"));
        assert!(bypass_proxy("anything", "*"));
    }
}
