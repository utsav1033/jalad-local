//! Where tokenpati keeps things, and what it remembers between runs.
//!
//! Everything lives under one folder so it is easy to find and easy to delete:
//!   $TOKENPATI_HOME, else ~/.tokenpati
//!     models/   GGUF and MLX weights we downloaded
//!     bin/      llama.cpp we installed
//!     logs/     engine server logs
//!     state.json settings + measured speeds

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

pub fn home() -> PathBuf {
    if let Some(h) = std::env::var_os("TOKENPATI_HOME") {
        return PathBuf::from(h);
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".tokenpati")
}

pub fn models_dir() -> PathBuf {
    home().join("models")
}

pub fn bin_dir() -> PathBuf {
    home().join("bin")
}

pub fn logs_dir() -> PathBuf {
    home().join("logs")
}

fn state_path() -> PathBuf {
    home().join("state.json")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// Context length to budget memory for and to load models with.
    pub context: u32,
    pub temperature: f32,
    pub system_prompt: String,
    /// "auto" | "ollama" | "llama.cpp" | "mlx"
    pub engine: String,
    /// First port tried for llama.cpp / mlx servers.
    pub port: u16,
    /// Show <think> blocks from reasoning models.
    pub show_thinking: bool,
    /// Scale predictions by what we measured on this machine.
    pub use_calibration: bool,
    pub mouse: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            context: 8192,
            temperature: 0.7,
            system_prompt: "You are a helpful assistant.".into(),
            engine: "auto".into(),
            port: 8686,
            show_thinking: true,
            use_calibration: true,
            mouse: true,
        }
    }
}

/// One real decode speed we saw, against what we predicted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Measurement {
    pub tok_s: f64,
    pub predicted: f64,
    pub engine: String,
    pub when: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct State {
    pub settings: Settings,
    /// Measured memory bandwidth, GB/s, from the built-in benchmark.
    pub measured_bandwidth: Option<f64>,
    /// "Model name|quant" -> latest measurement.
    pub measurements: BTreeMap<String, Measurement>,
}

impl State {
    pub fn load() -> State {
        std::fs::read_to_string(state_path())
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self) {
        let _ = std::fs::create_dir_all(home());
        if let Ok(s) = serde_json::to_string_pretty(self) {
            let tmp = state_path().with_extension("json.tmp");
            if std::fs::write(&tmp, s).is_ok() {
                let _ = std::fs::rename(tmp, state_path());
            }
        }
    }

    pub fn key(model: &str, quant: &str) -> String {
        format!("{model}|{quant}")
    }

    /// measured / predicted, averaged over everything we have measured.
    /// Prediction already includes EFFICIENCY, so 1.0 means the model is spot on.
    pub fn calibration(&self) -> Option<f64> {
        let ratios: Vec<f64> = self
            .measurements
            .values()
            .filter(|m| m.predicted > 0.0 && m.tok_s > 0.0)
            .map(|m| m.tok_s / m.predicted)
            .collect();
        if ratios.is_empty() {
            return None;
        }
        // Geometric mean: one 3x outlier should not dominate.
        let mean_ln = ratios.iter().map(|r| r.ln()).sum::<f64>() / ratios.len() as f64;
        Some(mean_ln.exp().clamp(0.2, 3.0))
    }
}

pub fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Free bytes on the filesystem holding `path` (or its nearest existing parent).
pub fn free_disk_bytes(path: &std::path::Path) -> Option<u64> {
    let mut p = path.to_path_buf();
    while !p.exists() {
        p = p.parent()?.to_path_buf();
    }
    let disks = sysinfo::Disks::new_with_refreshed_list();
    let canon = p.canonicalize().ok()?;
    disks
        .list()
        .iter()
        .filter(|d| canon.starts_with(d.mount_point()))
        .max_by_key(|d| d.mount_point().as_os_str().len())
        .map(|d| d.available_space())
}

pub fn human_bytes(b: u64) -> String {
    let b = b as f64;
    const K: f64 = 1024.0;
    if b >= K * K * K {
        format!("{:.1} GiB", b / (K * K * K))
    } else if b >= K * K {
        format!("{:.0} MiB", b / (K * K))
    } else if b >= K {
        format!("{:.0} KiB", b / K)
    } else {
        format!("{b:.0} B")
    }
}

/// Total size of every file under `path`.
pub fn dir_size(path: &std::path::Path) -> u64 {
    let Ok(rd) = std::fs::read_dir(path) else {
        return std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    };
    rd.flatten()
        .map(|e| {
            let p = e.path();
            if p.is_dir() {
                dir_size(&p)
            } else {
                e.metadata().map(|m| m.len()).unwrap_or(0)
            }
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_is_geometric_mean() {
        let mut s = State::default();
        assert!(s.calibration().is_none());
        for (i, (got, pred)) in [(20.0, 10.0), (5.0, 10.0)].iter().enumerate() {
            s.measurements.insert(
                format!("m{i}"),
                Measurement {
                    tok_s: *got,
                    predicted: *pred,
                    engine: "x".into(),
                    when: 0,
                },
            );
        }
        assert!((s.calibration().unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn bytes_read_like_people_talk() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(5 * 1024 * 1024), "5 MiB");
        assert_eq!(human_bytes(3 * 1024 * 1024 * 1024 / 2), "1.5 GiB");
    }
}
