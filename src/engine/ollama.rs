//! ollama, through its HTTP API on localhost:11434.

use super::{ChatOpts, Chunk, EngineInfo, EngineKind, Msg, Stats, ThinkSplitter};
use crate::net;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::BufRead;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// OLLAMA_HOST, normalised to a URL we can call.
pub fn host() -> String {
    let raw = std::env::var("OLLAMA_HOST").unwrap_or_default();
    let raw = raw.trim().trim_end_matches('/');
    if raw.is_empty() {
        return "http://127.0.0.1:11434".into();
    }
    let with_scheme = if raw.starts_with("http://") || raw.starts_with("https://") {
        raw.to_string()
    } else {
        format!("http://{raw}")
    };
    let fixed = with_scheme.replace("://0.0.0.0", "://127.0.0.1");
    // "http://host" with no port means the default port
    let after = fixed.split("://").nth(1).unwrap_or("");
    if after.contains(':') {
        fixed
    } else {
        format!("{fixed}:11434")
    }
}

pub fn binary() -> Option<PathBuf> {
    super::which("ollama").or_else(|| {
        let mac = PathBuf::from("/Applications/Ollama.app/Contents/Resources/ollama");
        mac.exists().then_some(mac)
    })
}

pub fn version(base: &str) -> Option<String> {
    let v: Value = net::local_agent()
        .get(&format!("{base}/api/version"))
        .timeout(Duration::from_millis(800))
        .call()
        .ok()?
        .into_json()
        .ok()?;
    v["version"].as_str().map(String::from)
}

pub fn detect() -> EngineInfo {
    let path = binary();
    let base = host();
    let running = version(&base);
    let installed = path.is_some() || running.is_some();
    EngineInfo {
        kind: EngineKind::Ollama,
        installed,
        running: running.is_some(),
        version: running.clone(),
        detail: match (&path, &running) {
            (_, Some(_)) => format!("server up at {base}"),
            (Some(_), None) => "installed, server not running (tokenpati starts it for you)".into(),
            (None, None) => "not installed".into(),
        },
        path,
    }
}

/// Start `ollama serve` in the background if nothing answers yet. Logs go to ~/.tokenpati/logs/ollama.log.
pub fn ensure_server(mut log: impl FnMut(String)) -> Result<String, String> {
    let base = host();
    if let Some(v) = version(&base) {
        return Ok(v);
    }
    let bin = binary()
        .ok_or("ollama is not installed. open the Setup tab (7) and press i, or use llama.cpp")?;
    log(format!("starting {} serve", bin.display()));
    let _ = std::fs::create_dir_all(crate::state::logs_dir());
    let logf = std::fs::File::create(crate::state::logs_dir().join("ollama.log"))
        .map_err(|e| e.to_string())?;
    let errf = logf.try_clone().map_err(|e| e.to_string())?;
    std::process::Command::new(bin)
        .arg("serve")
        .stdin(std::process::Stdio::null())
        .stdout(logf)
        .stderr(errf)
        .spawn()
        .map_err(|e| format!("could not start ollama: {e}"))?;
    for _ in 0..60 {
        std::thread::sleep(Duration::from_millis(250));
        if let Some(v) = version(&base) {
            log(format!("ollama {v} is up at {base}"));
            return Ok(v);
        }
    }
    Err(format!(
        "ollama did not come up at {base}. see {}",
        crate::state::logs_dir().join("ollama.log").display()
    ))
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct Details {
    #[serde(default)]
    pub parameter_size: String,
    #[serde(default)]
    pub quantization_level: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Tag {
    pub name: String,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub details: Details,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Loaded {
    pub name: String,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub size_vram: u64,
}

pub fn tags(base: &str) -> Result<Vec<Tag>, String> {
    let v: Value = net::local_agent()
        .get(&format!("{base}/api/tags"))
        .timeout(Duration::from_secs(3))
        .call()
        .map_err(net::err_text)?
        .into_json()
        .map_err(|e| e.to_string())?;
    serde_json::from_value(v["models"].clone()).map_err(|e| e.to_string())
}

pub fn ps(base: &str) -> Result<Vec<Loaded>, String> {
    let v: Value = net::local_agent()
        .get(&format!("{base}/api/ps"))
        .timeout(Duration::from_secs(3))
        .call()
        .map_err(net::err_text)?
        .into_json()
        .map_err(|e| e.to_string())?;
    serde_json::from_value(v["models"].clone()).map_err(|e| e.to_string())
}

pub fn has(base: &str, tag: &str) -> bool {
    let want = if tag.contains(':') {
        tag.to_string()
    } else {
        format!("{tag}:latest")
    };
    tags(base)
        .map(|ts| ts.iter().any(|t| t.name == want))
        .unwrap_or(false)
}

pub struct PullProgress {
    pub status: String,
    pub done: u64,
    pub total: u64,
}

/// `ollama pull`, streamed. Layers download in parallel, so progress is the sum over digests.
pub fn pull(
    base: &str,
    tag: &str,
    cancel: &Arc<AtomicBool>,
    mut on: impl FnMut(PullProgress),
) -> Result<(), String> {
    let resp = net::local_agent()
        .post(&format!("{base}/api/pull"))
        .send_json(json!({ "model": tag, "name": tag, "stream": true }))
        .map_err(net::err_text)?;
    let mut layers: HashMap<String, (u64, u64)> = HashMap::new();
    for line in std::io::BufReader::new(resp.into_reader()).lines() {
        if cancel.load(Ordering::Relaxed) {
            return Err(
                "cancelled. ollama keeps the partial layers, so pulling again resumes".into(),
            );
        }
        let line = line.map_err(|e| format!("pull interrupted: {e}"))?;
        let Ok(v) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        if let Some(err) = v["error"].as_str() {
            return Err(err.to_string());
        }
        let status = v["status"].as_str().unwrap_or("").to_string();
        if let (Some(d), Some(t)) = (v["digest"].as_str(), v["total"].as_u64()) {
            layers.insert(d.to_string(), (v["completed"].as_u64().unwrap_or(0), t));
        }
        let (done, total) = layers.values().fold((0, 0), |a, (c, t)| (a.0 + c, a.1 + t));
        on(PullProgress {
            status,
            done,
            total,
        });
    }
    Ok(())
}

/// Load into memory without generating, so the first chat message is not stuck behind the load.
pub fn load(base: &str, tag: &str, num_ctx: u32) -> Result<Duration, String> {
    let v: Value = net::local_agent()
        .post(&format!("{base}/api/generate"))
        .timeout(Duration::from_secs(900))
        .send_json(
            json!({ "model": tag, "prompt": "", "stream": false, "keep_alive": "30m",
                           "options": { "num_ctx": num_ctx } }),
        )
        .map_err(net::err_text)?
        .into_json()
        .map_err(|e| e.to_string())?;
    if let Some(e) = v["error"].as_str() {
        return Err(e.to_string());
    }
    Ok(Duration::from_nanos(
        v["load_duration"].as_u64().unwrap_or(0),
    ))
}

pub fn unload(base: &str, tag: &str) -> Result<(), String> {
    net::local_agent()
        .post(&format!("{base}/api/generate"))
        .timeout(Duration::from_secs(10))
        .send_json(json!({ "model": tag, "keep_alive": 0 }))
        .map_err(net::err_text)?;
    Ok(())
}

pub fn delete(base: &str, tag: &str) -> Result<(), String> {
    net::local_agent()
        .delete(&format!("{base}/api/delete"))
        .send_json(json!({ "model": tag, "name": tag }))
        .map_err(net::err_text)?;
    Ok(())
}

pub fn chat(
    base: &str,
    model: &str,
    msgs: &[Msg],
    opts: &ChatOpts,
    cancel: &Arc<AtomicBool>,
    mut on_chunk: impl FnMut(Chunk),
) -> Result<Stats, String> {
    let mut options = json!({ "num_ctx": opts.num_ctx, "temperature": opts.temperature });
    if let Some(n) = opts.max_tokens {
        options["num_predict"] = json!(n);
    }
    let started = Instant::now();
    let resp = net::local_agent()
        .post(&format!("{base}/api/chat"))
        .send_json(json!({ "model": model, "messages": msgs, "stream": true, "keep_alive": "30m", "options": options }))
        .map_err(net::err_text)?;
    let mut first: Option<Instant> = None;
    let mut chunks = 0u64;
    let mut think = ThinkSplitter::default();
    for line in std::io::BufReader::new(resp.into_reader()).lines() {
        if cancel.load(Ordering::Relaxed) {
            break; // dropping the reader closes the connection, which stops generation
        }
        let line = line.map_err(|e| e.to_string())?;
        let Ok(v) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        if let Some(e) = v["error"].as_str() {
            return Err(e.to_string());
        }
        if let Some(t) = v["message"]["thinking"].as_str().filter(|t| !t.is_empty()) {
            first.get_or_insert_with(Instant::now);
            on_chunk(Chunk::Thinking(t.to_string()));
        }
        if let Some(t) = v["message"]["content"].as_str().filter(|t| !t.is_empty()) {
            first.get_or_insert_with(Instant::now);
            chunks += 1;
            think.push(t, &mut on_chunk);
        }
        if v["done"].as_bool() == Some(true) {
            think.finish(&mut on_chunk);
            let eval_n = v["eval_count"].as_u64().unwrap_or(chunks);
            let eval_ns = v["eval_duration"].as_u64().unwrap_or(0);
            let p_n = v["prompt_eval_count"].as_u64();
            let p_ns = v["prompt_eval_duration"].as_u64().unwrap_or(0);
            return Ok(Stats {
                tokens: eval_n,
                tok_s: if eval_ns > 0 {
                    eval_n as f64 / (eval_ns as f64 / 1e9)
                } else {
                    0.0
                },
                ttft_ms: first.map(|f| (f - started).as_millis() as u64).unwrap_or(0),
                prompt_tokens: p_n,
                prompt_tok_s: p_n
                    .filter(|_| p_ns > 0)
                    .map(|n| n as f64 / (p_ns as f64 / 1e9)),
                exact: eval_ns > 0,
            });
        }
    }
    think.finish(&mut on_chunk);
    // Stopped early: estimate from our own clock.
    let gen_secs = first.map(|f| f.elapsed().as_secs_f64()).unwrap_or(0.0);
    Ok(Stats {
        tokens: chunks,
        tok_s: if gen_secs > 0.0 {
            chunks as f64 / gen_secs
        } else {
            0.0
        },
        ttft_ms: first.map(|f| (f - started).as_millis() as u64).unwrap_or(0),
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_normalises() {
        std::env::remove_var("OLLAMA_HOST");
        assert_eq!(host(), "http://127.0.0.1:11434");
        std::env::set_var("OLLAMA_HOST", "0.0.0.0");
        assert_eq!(host(), "http://127.0.0.1:11434");
        std::env::set_var("OLLAMA_HOST", "box.local:9999/");
        assert_eq!(host(), "http://box.local:9999");
        std::env::remove_var("OLLAMA_HOST");
    }
}
