//! The maths. Units: GiB everywhere, GB/s for bandwidth.
//!
//! weights = params * bytes_per_param(quant) * OVERHEAD
//! kv      = 2 * layers * kv_heads * head_dim * context * kv_bytes
//! tok/s   = bandwidth / (active_weights + kv(SPEED_CONTEXT)) * EFFICIENCY
//!           decode is memory-bound: every token re-reads the active weights plus the filled KV.
//!           Speed is quoted at SPEED_CONTEXT (a typical chat); memory is budgeted at target_context.
//! fits    = weights + kv(target_context) <= hw.usable_gib

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
    hw.bandwidth_gbps * 1e9 / bytes_per_token * EFFICIENCY
}

pub fn max_context_that_fits(hw: &Hardware, model: &Model, quant: Quant) -> u32 {
    let w = weight_gib(model, quant);
    let mut best = 0;
    let mut ctx = 1024;
    while ctx <= model.max_context {
        if w + kv_gib(model, ctx) <= hw.usable_gib {
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
    Estimate {
        model: *model,
        quant,
        weight_gib: w,
        kv_gib: kv,
        total_gib: total,
        fits: total <= hw.usable_gib,
        tok_s: predicted_tok_s(hw, active_weight_gib(model, quant), kv_gib(model, SPEED_CONTEXT.min(target_context))),
        max_context: max_context_that_fits(hw, model, quant).min(model.max_context),
        target_context,
    }
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
    fn ollama_tags() {
        assert_eq!(Quant::Q4KM.ollama_tag(), "q4_K_M");
        assert_eq!(Quant::Q8_0.ollama_tag(), "q8_0");
        assert_eq!(Quant::FP16.ollama_tag(), "fp16");
    }
}
