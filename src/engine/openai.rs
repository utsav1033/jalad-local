//! /v1/chat/completions with server-sent events. llama.cpp and mlx both speak it.

use super::{ChatOpts, Chunk, Msg, Stats, ThinkSplitter};
use crate::net;
use serde_json::{json, Value};
use std::io::BufRead;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub fn chat(
    base: &str,
    model: &str,
    msgs: &[Msg],
    opts: &ChatOpts,
    cancel: &Arc<AtomicBool>,
    mut on_chunk: impl FnMut(Chunk),
) -> Result<Stats, String> {
    let mut body = json!({
        "model": model,
        "messages": msgs,
        "stream": true,
        "temperature": opts.temperature,
        "stream_options": { "include_usage": true },
    });
    if let Some(n) = opts.max_tokens {
        body["max_tokens"] = json!(n);
    }
    let started = Instant::now();
    let resp = net::local_agent()
        .post(&format!("{base}/v1/chat/completions"))
        .send_json(body)
        .map_err(net::err_text)?;

    let mut first: Option<Instant> = None;
    let mut chunks = 0u64;
    let mut usage_tokens: Option<u64> = None;
    let mut timings: Option<Value> = None;
    let mut think = ThinkSplitter::default();
    for line in std::io::BufReader::new(resp.into_reader()).lines() {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let line = line.map_err(|e| e.to_string())?;
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data == "[DONE]" {
            break;
        }
        let Ok(v) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        if let Some(e) = v.get("error") {
            return Err(e["message"].as_str().unwrap_or(&e.to_string()).to_string());
        }
        let delta = &v["choices"][0]["delta"];
        for key in ["reasoning_content", "reasoning"] {
            if let Some(t) = delta[key].as_str().filter(|t| !t.is_empty()) {
                first.get_or_insert_with(Instant::now);
                chunks += 1;
                on_chunk(Chunk::Thinking(t.to_string()));
            }
        }
        if let Some(t) = delta["content"].as_str().filter(|t| !t.is_empty()) {
            first.get_or_insert_with(Instant::now);
            chunks += 1;
            think.push(t, &mut on_chunk);
        }
        if let Some(n) = v["usage"]["completion_tokens"].as_u64() {
            usage_tokens = Some(n);
        }
        if v.get("timings").is_some() {
            timings = Some(v["timings"].clone());
        }
    }
    think.finish(&mut on_chunk);

    let ttft_ms = first.map(|f| (f - started).as_millis() as u64).unwrap_or(0);
    // llama.cpp reports its own timings; trust them over our stopwatch.
    if let Some(t) = timings {
        if let (Some(n), Some(tps)) = (
            t["predicted_n"].as_u64(),
            t["predicted_per_second"].as_f64(),
        ) {
            return Ok(Stats {
                tokens: n,
                tok_s: tps,
                ttft_ms,
                prompt_tokens: t["prompt_n"].as_u64(),
                prompt_tok_s: t["prompt_per_second"].as_f64(),
                exact: true,
            });
        }
    }
    let tokens = usage_tokens.unwrap_or(chunks);
    // Decode speed excludes time to first token (that is prompt processing).
    let gen_secs = first.map(|f| f.elapsed().as_secs_f64()).unwrap_or(0.0);
    Ok(Stats {
        tokens,
        tok_s: if gen_secs > 0.05 && tokens > 1 {
            (tokens - 1) as f64 / gen_secs
        } else {
            0.0
        },
        ttft_ms,
        prompt_tokens: None,
        prompt_tok_s: None,
        exact: false,
    })
}

/// Poll until the server answers `path` with 200, or give up.
/// `alive` lets the caller bail out early when the server process has already died.
pub fn wait_ready(
    base: &str,
    path: &str,
    timeout: Duration,
    cancel: &Arc<AtomicBool>,
    mut alive: impl FnMut() -> Result<(), String>,
    mut tick: impl FnMut(Duration),
) -> Result<Duration, String> {
    let start = Instant::now();
    let agent = net::local_agent();
    while start.elapsed() < timeout {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        alive()?;
        if let Ok(r) = agent
            .get(&format!("{base}{path}"))
            .timeout(Duration::from_secs(2))
            .call()
        {
            if r.status() == 200 {
                return Ok(start.elapsed());
            }
        }
        tick(start.elapsed());
        std::thread::sleep(Duration::from_millis(300));
    }
    Err(format!(
        "server did not become ready within {}s",
        timeout.as_secs()
    ))
}
