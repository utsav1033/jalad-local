//! The maths. Units: GiB everywhere, GB/s for bandwidth.
//!
//! weights = params * bytes_per_param(quant) * OVERHEAD
//! kv      = 2 * layers * kv_heads * head_dim * context * kv_bytes
//! tok/s   = bandwidth / (active_weights + kv(SPEED_CONTEXT)) * efficiency (EFFICIENCY, calibrated by measurements)
//!           decode is memory-bound: every token re-reads the active weights plus the filled KV.
//!           Speed is quoted at SPEED_CONTEXT (a typical chat); memory is budgeted at target_context.
//! fits    = weights + kv(target_context) <= hw.usable_gib
//!
//! Partial offload (discrete GPU only): a model bigger than VRAM can still run with llama.cpp by keeping
//! some layers in system RAM. Each token then reads the GPU layers at GPU bandwidth and the rest at RAM
//! bandwidth, so time per token is the sum of the two reads.

use crate::catalog::Model;
use crate::hardware::Hardware;
use serde::Serialize;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
/// Embeddings, runtime buffers, scratch.
pub const OVERHEAD: f64 = 1.10;
/// Fraction of peak bandwidth real decode achieves. Measure and correct.
pub const EFFICIENCY: f64 = 0.80;
/// fp16 KV cache.
pub const KV_BYTES: f64 = 2.0;
/// Context fill assumed when quoting tok/s.
pub const SPEED_CONTEXT: u32 = 2048;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum Quant {
    Q4KM,
    Q5KM,
    Q6K,
    Q8_0,
    FP16,
}

impl Quant {
    pub const ALL: [Quant; 5] = [Quant::Q4KM, Quant::Q5KM, Quant::Q6K, Quant::Q8_0, Quant::FP16];

    /// K-quants sit above their nominal bits because some tensors stay at higher precision.
    pub fn bytes_per_param(self) -> f64 {
        match self {
            Quant::Q4KM => 0.56,
            Quant::Q5KM => 0.68,
            Quant::Q6K => 0.80,
            Quant::Q8_0 => 1.06,
            Quant::FP16 => 2.0,
        }
    }

    pub fn tag(self) -> &'static str {
        match self {
            Quant::Q4KM => "Q4_K_M",
            Quant::Q5KM => "Q5_K_M",
            Quant::Q6K => "Q6_K",
            Quant::Q8_0 => "Q8_0",
            Quant::FP16 => "FP16",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Quant::Q4KM => "4-bit",
            Quant::Q5KM => "5-bit",
            Quant::Q6K => "6-bit",
            Quant::Q8_0 => "8-bit",
            Quant::FP16 => "fp16",
        }
    }

    /// ollama's tag spelling: Q4_K_M -> q4_K_M, FP16 -> fp16.
    pub fn ollama_tag(self) -> String {
        match self {
            Quant::FP16 => "fp16".into(),
            q => {
                let t = q.tag();
                format!("{}{}", t[..1].to_lowercase(), &t[1..])
            }
        }
    }

    /// No 5-bit on mlx.
    pub fn mlx_tag(self) -> Option<&'static str> {
        match self {
            Quant::Q4KM => Some("4bit"),
            Quant::Q6K => Some("6bit"),
            Quant::Q8_0 => Some("8bit"),
            Quant::FP16 => Some("bf16"),
            Quant::Q5KM => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Estimate {
    pub model: Model,
    /// Layers on the GPU (llama.cpp -ngl). None where the whole model shares one memory pool.
    pub gpu_layers: Option<u32>,
    /// True when some layers live in system RAM.
    pub offload: bool,
    pub quant: Quant,
    pub weight_gib: f64,
    /// At target context.
    pub kv_gib: f64,
    pub total_gib: f64,
    pub fits: bool,
    pub tok_s: f64,
    pub max_context: u32,
    pub target_context: u32,
}

pub fn weight_gib(model: &Model, quant: Quant) -> f64 {
    model.params_b * 1e9 * quant.bytes_per_param() * OVERHEAD / GIB
}

pub fn active_weight_gib(model: &Model, quant: Quant) -> f64 {
    model.active_b * 1e9 * quant.bytes_per_param() * OVERHEAD / GIB
}

pub fn kv_gib(model: &Model, context: u32) -> f64 {
    2.0 * model.layers as f64 * model.kv_heads as f64 * model.head_dim as f64 * context as f64 * KV_BYTES / GIB
}

pub fn predicted_tok_s(hw: &Hardware, active_gib: f64, kv: f64) -> f64 {
    let bytes_per_token = (active_gib + kv) * GIB;
    hw.bandwidth_gbps * 1e9 / bytes_per_token * hw.efficiency
}

/// How many of `layers` fit in VRAM when the whole thing needs `total` GiB.
pub fn gpu_layers_that_fit(hw: &Hardware, layers: u32, total: f64) -> u32 {
    if total <= hw.usable_gib {
        return layers;
    }
    let per_layer = total / layers.max(1) as f64;
    ((hw.usable_gib / per_layer).floor() as u32).min(layers)
}

/// Time per token when `gpu_frac` of the bytes sit in VRAM and the rest in RAM.
pub fn offload_tok_s(hw: &Hardware, bytes_gib: f64, gpu_frac: f64) -> f64 {
    let gpu = bytes_gib * gpu_frac * GIB / (hw.bandwidth_gbps * 1e9);
    let cpu = bytes_gib * (1.0 - gpu_frac) * GIB / (hw.ram_bandwidth_gbps * 1e9);
    hw.efficiency / (gpu + cpu)
}

pub fn max_context_that_fits(hw: &Hardware, model: &Model, quant: Quant) -> u32 {
    let w = weight_gib(model, quant);
    let mut best = 0;
    let mut ctx = 1024;
    while ctx <= model.max_context {
        if w + kv_gib(model, ctx) <= hw.capacity_gib() {
            best = ctx;
        }
        ctx *= 2;
    }
    best
}

pub fn estimate(hw: &Hardware, model: &Model, quant: Quant, target_context: u32) -> Estimate {
    let w = weight_gib(model, quant);
    let kv = kv_gib(model, target_context);
    let total = w + kv;
    let in_vram = total <= hw.usable_gib;
    let fits = total <= hw.capacity_gib();
    let speed_kv = kv_gib(model, SPEED_CONTEXT.min(target_context));
    let active = active_weight_gib(model, quant);
    let (tok_s, gpu_layers, offload) = if !hw.has_discrete_gpu() {
        (predicted_tok_s(hw, active, speed_kv), None, false)
    } else if in_vram {
        (predicted_tok_s(hw, active, speed_kv), Some(model.layers), false)
    } else {
        let ngl = gpu_layers_that_fit(hw, model.layers, total);
        let frac = ngl as f64 / model.layers as f64;
        (offload_tok_s(hw, active + speed_kv, frac), Some(ngl), true)
    };
    Estimate {
        model: *model,
        gpu_layers,
        offload: offload && fits,
        quant,
        weight_gib: w,
        kv_gib: kv,
        total_gib: total,
        fits,
        tok_s,
        max_context: max_context_that_fits(hw, model, quant).min(model.max_context),
        target_context,
    }
}

/// Every quant of one model, smallest first. What the detail pane shows.
pub fn quant_matrix(hw: &Hardware, model: &Model, target_context: u32) -> Vec<Estimate> {
    Quant::ALL
        .iter()
        .filter(|q| !hw.is_apple() || q.mlx_tag().is_some())
        .map(|q| estimate(hw, model, *q, target_context))
        .collect()
}

/// Guess the quant from a GGUF filename like "Qwen3-8B-Q5_K_M.gguf".
pub fn quant_from_filename(name: &str) -> Option<String> {
    let up = name.to_uppercase();
    const KNOWN: &[&str] = &[
        "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M", "IQ4_XS",
        "IQ4_NL", "Q2_K_L", "Q2_K", "Q3_K_XL", "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q4_K_L", "Q4_K_M", "Q4_K_S", "Q4_0",
        "Q4_1", "Q5_K_L", "Q5_K_M", "Q5_K_S", "Q5_0", "Q5_1", "Q6_K_L", "Q6_K", "Q8_0", "BF16", "F16", "F32",
        "MXFP4",
    ];
    KNOWN.iter().find(|k| up.contains(*k)).map(|k| k.to_string())
}

pub fn estimate_all(hw: &Hardware, catalog: &[Model], target_context: u32) -> Vec<Estimate> {
    let quants: Vec<Quant> = Quant::ALL
        .iter()
        .copied()
        .filter(|q| !hw.is_apple() || q.mlx_tag().is_some())
        .collect();
    catalog
        .iter()
        .flat_map(|m| quants.iter().map(move |q| estimate(hw, m, *q, target_context)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::CATALOG;

    fn llama_8b() -> Model {
        *CATALOG.iter().find(|m| m.name == "Llama 3.1 8B").unwrap()
    }

    #[test]
    fn kv_cache_llama_8b_at_8k_is_one_gib() {
        // 2 * 32 * 8 * 128 * 8192 * 2 bytes = 1 GiB exactly
        assert!((kv_gib(&llama_8b(), 8192) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn weight_q4_llama_8b() {
        // 8.03e9 * 0.56 * 1.1 / 2^30 = 4.61 GiB; the real Q4_K_M GGUF is 4.58 GiB
        assert!((weight_gib(&llama_8b(), Quant::Q4KM) - 4.6).abs() < 0.15);
    }

    #[test]
    fn offload_runs_big_models_slower() {
        let hw = Hardware {
            chip: "x".into(), os: "linux".into(), memory_gib: 64.0, usable_gib: 23.0,
            bandwidth_gbps: 1000.0, bandwidth_estimated: false, backend: "llama.cpp-cuda".into(),
            vram_gib: Some(24.0), gpu_name: Some("RTX 4090".into()), gpu_cores: None,
            offload_gib: 60.0, ram_bandwidth_gbps: 60.0, efficiency: EFFICIENCY, cpu_threads: 16,
        };
        let big = *CATALOG.iter().find(|m| m.name == "Llama 3.3 70B").unwrap();
        let e = estimate(&hw, &big, Quant::Q4KM, 8192);
        assert!(e.fits && e.offload);
        let ngl = e.gpu_layers.unwrap();
        assert!(ngl > 0 && ngl < big.layers);
        // slower than all-GPU, faster than all-CPU
        assert!(e.tok_s < predicted_tok_s(&hw, active_weight_gib(&big, Quant::Q4KM), 0.0));
        assert!(e.tok_s > offload_tok_s(&hw, active_weight_gib(&big, Quant::Q4KM), 0.0));
        let small = llama_8b();
        let e = estimate(&hw, &small, Quant::Q8_0, 8192);
        assert!(!e.offload && e.gpu_layers == Some(small.layers));
    }

    #[test]
    fn quant_from_names() {
        assert_eq!(quant_from_filename("Qwen_Qwen3-8B-Q5_K_M.gguf").as_deref(), Some("Q5_K_M"));
        assert_eq!(quant_from_filename("x-IQ4_XS.gguf").as_deref(), Some("IQ4_XS"));
        assert_eq!(quant_from_filename("model-q8_0.gguf").as_deref(), Some("Q8_0"));
        assert_eq!(quant_from_filename("model.gguf"), None);
    }

    #[test]
    fn ollama_tags() {
        assert_eq!(Quant::Q4KM.ollama_tag(), "q4_K_M");
        assert_eq!(Quant::Q8_0.ollama_tag(), "q8_0");
        assert_eq!(Quant::FP16.ollama_tag(), "fp16");
    }
}
