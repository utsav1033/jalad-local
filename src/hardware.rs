//! What machine is this: chip, memory, bandwidth, backend.

use serde::Serialize;
use std::process::Command;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Memory bandwidth in GB/s from Apple's spec pages. Longest matching key wins.
/// Chips that ship in two bins (M3 Max 300/400, M4 Max 410/546) use the lower bin.
const APPLE_BANDWIDTH: &[(&str, f64)] = &[
    ("M1", 68.25), ("M1 Pro", 200.0), ("M1 Max", 400.0), ("M1 Ultra", 800.0),
    ("M2", 100.0), ("M2 Pro", 200.0), ("M2 Max", 400.0), ("M2 Ultra", 800.0),
    ("M3", 100.0), ("M3 Pro", 150.0), ("M3 Max", 300.0), ("M3 Ultra", 800.0),
    ("M4", 120.0), ("M4 Pro", 273.0), ("M4 Max", 410.0),
    ("M5", 153.0), ("M5 Pro", 300.0), ("M5 Max", 450.0),
];

const NVIDIA_BANDWIDTH: &[(&str, f64)] = &[
    ("H100", 3350.0), ("A100", 1935.0), ("L40S", 864.0), ("A10", 600.0), ("T4", 320.0),
    ("RTX 6000 Ada", 960.0), ("RTX A6000", 768.0),
    ("RTX 5090", 1792.0), ("RTX 5080", 960.0), ("RTX 5070 Ti", 896.0), ("RTX 5070", 672.0),
    ("RTX 4090", 1008.0), ("RTX 4080", 717.0), ("RTX 4070 Ti", 504.0), ("RTX 4070", 504.0),
    ("RTX 4060 Ti", 288.0), ("RTX 4060", 272.0),
    ("RTX 3090", 936.0), ("RTX 3080", 760.0), ("RTX 3070", 448.0), ("RTX 3060", 360.0),
];

/// Apple caps how much unified memory the GPU may wire by default (roughly 75%).
const APPLE_GPU_SHARE: f64 = 0.75;
const CUDA_RESERVE_GIB: f64 = 1.0;
const CPU_RESERVE_GIB: f64 = 4.0;
const UNKNOWN_BANDWIDTH: f64 = 60.0;

#[derive(Debug, Clone, Serialize)]
pub struct Hardware {
    pub chip: String,
    pub os: String,
    pub memory_gib: f64,
    /// What a model may actually occupy.
    pub usable_gib: f64,
    pub bandwidth_gbps: f64,
    pub bandwidth_estimated: bool,
    /// "mlx" | "llama.cpp-cuda" | "llama.cpp-cpu"
    pub backend: String,
    pub vram_gib: Option<f64>,
    pub gpu_name: Option<String>,
    pub gpu_cores: Option<u32>,
    /// System RAM a GPU build may spill layers into (partial offload). 0 when there is no discrete GPU.
    pub offload_gib: f64,
    /// Bandwidth of that system RAM, GB/s.
    pub ram_bandwidth_gbps: f64,
    /// Fraction of peak bandwidth decode achieves. Starts at budget::EFFICIENCY, calibrated by measurements.
    pub efficiency: f64,
    pub cpu_threads: usize,
}

impl Hardware {
    pub fn is_apple(&self) -> bool {
        self.backend == "mlx"
    }

    /// Everything a model may occupy, counting RAM spill on discrete GPUs.
    pub fn capacity_gib(&self) -> f64 {
        self.usable_gib + self.offload_gib
    }

    pub fn has_discrete_gpu(&self) -> bool {
        self.vram_gib.is_some()
    }

    /// Fold in what we measured on this machine before: real bandwidth and real tok/s.
    pub fn apply_state(&mut self, state: &crate::state::State) {
        if let Some(bw) = state.measured_bandwidth {
            if self.has_discrete_gpu() {
                self.ram_bandwidth_gbps = bw;
            } else if self.bandwidth_estimated {
                self.bandwidth_gbps = bw;
                self.bandwidth_estimated = false;
            }
        }
        if state.settings.use_calibration {
            if let Some(c) = state.calibration() {
                self.efficiency = (crate::budget::EFFICIENCY * c).clamp(0.1, 1.0);
            }
        }
    }

    pub fn memory_label(&self) -> String {
        match self.vram_gib {
            Some(v) => format!("{:.0} GiB VRAM  (+{:.0} GiB RAM)", v, self.memory_gib),
            None if self.is_apple() => format!("{:.0} GiB unified", self.memory_gib),
            None => format!("{:.0} GiB RAM", self.memory_gib),
        }
    }
}

fn run(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn sysctl(key: &str) -> Option<String> {
    run("sysctl", &["-n", key])
}

/// Longest key that appears in `name` on word boundaries.
fn lookup(table: &[(&str, f64)], name: &str) -> Option<f64> {
    let mut keys: Vec<&(&str, f64)> = table.iter().collect();
    keys.sort_by_key(|(k, _)| std::cmp::Reverse(k.len()));
    let bytes = name.as_bytes();
    for (key, bw) in keys {
        let mut start = 0;
        while let Some(pos) = name[start..].find(key) {
            let i = start + pos;
            let j = i + key.len();
            let before_ok = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();
            let after_ok = j == bytes.len() || !bytes[j].is_ascii_alphanumeric();
            if before_ok && after_ok {
                return Some(*bw);
            }
            start = i + 1;
        }
    }
    None
}

fn apple_gpu_cores() -> Option<u32> {
    let out = run("system_profiler", &["SPDisplaysDataType"])?;
    out.lines()
        .find_map(|l| l.trim().strip_prefix("Total Number of Cores:"))
        .and_then(|v| v.trim().parse().ok())
}

fn nvidia() -> Option<(String, f64)> {
    let out = run(
        "nvidia-smi",
        &["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
    )?;
    let first = out.lines().next()?;
    let mut parts = first.split(',').map(|s| s.trim());
    let name = parts.next()?.to_string();
    let mib: f64 = parts.next()?.parse().ok()?;
    Some((name, mib / 1024.0))
}

fn linux_cpu_name() -> Option<String> {
    let s = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    s.lines()
        .find(|l| l.starts_with("model name"))
        .and_then(|l| l.split(':').nth(1))
        .map(|v| v.trim().to_string())
}

pub fn detect() -> Hardware {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let memory_gib = sys.total_memory() as f64 / GIB;
    let os = std::env::consts::OS.to_string();

    if os == "macos" {
        let chip = sysctl("machdep.cpu.brand_string").unwrap_or_else(|| "unknown".into());
        if chip.contains("Apple") {
            let bw = lookup(APPLE_BANDWIDTH, &chip);
            return Hardware {
                chip,
                os,
                memory_gib,
                usable_gib: memory_gib * APPLE_GPU_SHARE,
                bandwidth_gbps: bw.unwrap_or(UNKNOWN_BANDWIDTH),
                bandwidth_estimated: bw.is_none(),
                backend: "mlx".into(),
                vram_gib: None,
                gpu_name: None,
                gpu_cores: apple_gpu_cores(),
                offload_gib: 0.0,
                ram_bandwidth_gbps: bw.unwrap_or(UNKNOWN_BANDWIDTH),
                efficiency: crate::budget::EFFICIENCY,
                cpu_threads: threads(),
            };
        }
        return cpu_only(chip, os, memory_gib);
    }

    let chip = linux_cpu_name().unwrap_or_else(|| std::env::consts::ARCH.to_string());
    if let Some((name, vram)) = nvidia() {
        let bw = lookup(NVIDIA_BANDWIDTH, &name);
        return Hardware {
            chip,
            os,
            memory_gib,
            usable_gib: (vram - CUDA_RESERVE_GIB).max(1.0),
            bandwidth_gbps: bw.unwrap_or(400.0),
            bandwidth_estimated: bw.is_none(),
            backend: "llama.cpp-cuda".into(),
            vram_gib: Some(vram),
            gpu_name: Some(name),
            gpu_cores: None,
            offload_gib: (memory_gib - CPU_RESERVE_GIB).max(0.0),
            ram_bandwidth_gbps: UNKNOWN_BANDWIDTH,
            efficiency: crate::budget::EFFICIENCY,
            cpu_threads: threads(),
        };
    }
    cpu_only(chip, os, memory_gib)
}

fn cpu_only(chip: String, os: String, memory_gib: f64) -> Hardware {
    Hardware {
        chip,
        os,
        memory_gib,
        usable_gib: (memory_gib - CPU_RESERVE_GIB).max(1.0),
        bandwidth_gbps: UNKNOWN_BANDWIDTH,
        bandwidth_estimated: true,
        backend: "llama.cpp-cpu".into(),
        vram_gib: None,
        gpu_name: None,
        gpu_cores: None,
        offload_gib: 0.0,
        ram_bandwidth_gbps: UNKNOWN_BANDWIDTH,
        efficiency: crate::budget::EFFICIENCY,
        cpu_threads: threads(),
    }
}

fn threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

/// Read a buffer much larger than any cache from every core at once and time it.
/// This is the number that decides tok/s on a CPU, so measuring beats guessing.
pub fn bench_bandwidth() -> f64 {
    const PER_THREAD: usize = 64 * 1024 * 1024; // 64 MiB of u64 per thread
    let n = threads().clamp(1, 32);
    let bufs: Vec<Vec<u64>> = (0..n).map(|i| vec![i as u64 + 1; PER_THREAD / 8]).collect();
    let mut best = 0.0f64;
    for _ in 0..3 {
        let start = std::time::Instant::now();
        let total: u64 = std::thread::scope(|sc| {
            let hs: Vec<_> = bufs
                .iter()
                .map(|b| {
                    sc.spawn(move || {
                        // Four accumulators so the loop is limited by memory, not by the add chain.
                        let (mut a, mut c, mut d, mut e) = (0u64, 0u64, 0u64, 0u64);
                        for ch in b.chunks_exact(4) {
                            a = a.wrapping_add(ch[0]);
                            c = c.wrapping_add(ch[1]);
                            d = d.wrapping_add(ch[2]);
                            e = e.wrapping_add(ch[3]);
                        }
                        std::hint::black_box(a ^ c ^ d ^ e)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap_or(0)).fold(0, |x, y| x ^ y)
        });
        std::hint::black_box(total);
        let secs = start.elapsed().as_secs_f64();
        best = best.max((n * PER_THREAD) as f64 / secs / 1e9);
    }
    best
}

/// What the machine is doing right now.
#[derive(Debug, Clone, Default)]
pub struct Live {
    pub ram_used_gib: f64,
    pub ram_total_gib: f64,
    pub cpu_pct: f32,
    pub gpu_util_pct: Option<f32>,
    pub vram_used_gib: Option<f64>,
    pub vram_total_gib: Option<f64>,
    /// Resident memory of the model server tokenpati started, if any.
    pub model_rss_gib: Option<f64>,
}

pub struct Sampler {
    sys: sysinfo::System,
    nvidia: bool,
}

impl Sampler {
    pub fn new(hw: &Hardware) -> Self {
        let mut sys = sysinfo::System::new();
        sys.refresh_cpu_usage();
        sys.refresh_memory();
        Sampler { sys, nvidia: hw.gpu_name.is_some() && hw.backend.contains("cuda") }
    }

    pub fn sample(&mut self) -> Live {
        self.sys.refresh_cpu_usage();
        self.sys.refresh_memory();
        let mut live = Live {
            ram_used_gib: self.sys.used_memory() as f64 / GIB,
            ram_total_gib: self.sys.total_memory() as f64 / GIB,
            cpu_pct: self.sys.global_cpu_usage(),
            ..Default::default()
        };
        if self.nvidia {
            let out = run("nvidia-smi", &["--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]);
            let v: Vec<f64> = out
                .as_deref()
                .and_then(|o| o.lines().next())
                .unwrap_or("")
                .split(',')
                .filter_map(|x| x.trim().parse().ok())
                .collect();
            if v.len() == 3 {
                live.gpu_util_pct = Some(v[0] as f32);
                live.vram_used_gib = Some(v[1] / 1024.0);
                live.vram_total_gib = Some(v[2] / 1024.0);
            }
        }
        live
    }
}

/// Resident memory of a process and its children, GiB. Used to show what a loaded model really costs.
pub fn process_rss_gib(pid: u32) -> Option<f64> {
    let mut sys = sysinfo::System::new();
    sys.refresh_processes(sysinfo::ProcessesToUpdate::All, true);
    let root = sysinfo::Pid::from_u32(pid);
    let mut total = 0u64;
    let mut found = false;
    for (p, proc_) in sys.processes() {
        if *p == root || proc_.parent() == Some(root) {
            total += proc_.memory();
            found = true;
        }
    }
    found.then(|| total as f64 / GIB)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn longest_key_wins() {
        assert_eq!(lookup(APPLE_BANDWIDTH, "Apple M3 Max"), Some(300.0));
        assert_eq!(lookup(APPLE_BANDWIDTH, "Apple M3"), Some(100.0));
        assert_eq!(lookup(NVIDIA_BANDWIDTH, "NVIDIA GeForce RTX 4070 Ti"), Some(504.0));
        assert_eq!(lookup(APPLE_BANDWIDTH, "Intel i9"), None);
    }
}
