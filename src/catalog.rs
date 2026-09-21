//! The contestants. Architecture numbers from each model's config.json.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Model {
    pub name: &'static str,
    /// Total parameters, billions (memory).
    pub params_b: f64,
    /// Parameters touched per token (speed); equals params_b unless MoE.
    pub active_b: f64,
    pub layers: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub max_context: u32,
    /// Base ollama tag, quant suffix appended.
    pub ollama: &'static str,
    /// mlx-community repo stem, "-4bit" etc appended.
    pub mlx: &'static str,
    pub tagline: &'static str,
}

impl Model {
    pub fn is_moe(&self) -> bool {
        self.active_b < self.params_b
    }
}

macro_rules! m {
    ($name:expr, $p:expr, $a:expr, $l:expr, $kv:expr, $hd:expr, $ctx:expr, $ollama:expr, $mlx:expr, $tag:expr) => {
        Model { name: $name, params_b: $p, active_b: $a, layers: $l, kv_heads: $kv, head_dim: $hd,
                max_context: $ctx, ollama: $ollama, mlx: $mlx, tagline: $tag }
    };
}

pub const CATALOG: &[Model] = &[
    m!("Llama 3.2 1B", 1.24, 1.24, 16, 8, 64, 131072, "llama3.2:1b-instruct", "Llama-3.2-1B-Instruct", "tiny, for autocomplete and toys"),
    m!("Llama 3.2 3B", 3.21, 3.21, 28, 8, 128, 131072, "llama3.2:3b-instruct", "Llama-3.2-3B-Instruct", "small but coherent"),
    m!("Gemma 3 4B", 4.3, 4.3, 34, 4, 256, 131072, "gemma3:4b-it", "gemma-3-4b-it", "small, sees images"),
    m!("Qwen2.5 7B", 7.6, 7.6, 28, 4, 128, 32768, "qwen2.5:7b-instruct", "Qwen2.5-7B-Instruct", "the reliable 7B"),
    m!("Llama 3.1 8B", 8.03, 8.03, 32, 8, 128, 131072, "llama3.1:8b-instruct", "Meta-Llama-3.1-8B-Instruct", "the default everyone benchmarks"),
    m!("Qwen3 8B", 8.2, 8.2, 36, 8, 128, 40960, "qwen3:8b", "Qwen3-8B", "thinking mode, the new default 8B"),
    m!("Gemma 3 12B", 12.2, 12.2, 48, 8, 256, 131072, "gemma3:12b-it", "gemma-3-12b-it", "strong writer, sees images"),
    m!("Qwen3 14B", 14.8, 14.8, 40, 8, 128, 40960, "qwen3:14b", "Qwen3-14B", "thinking mode, good at code"),
    m!("Phi-4 14B", 14.7, 14.7, 40, 10, 128, 16384, "phi4:14b", "phi-4", "punches above its weight on reasoning"),
    m!("Mistral Small 3.1 24B", 24.0, 24.0, 40, 8, 128, 131072, "mistral-small3.1:24b-instruct-2503", "Mistral-Small-3.1-24B-Instruct-2503", "the sweet spot if it fits"),
    m!("Gemma 3 27B", 27.4, 27.4, 62, 16, 128, 131072, "gemma3:27b-it", "gemma-3-27b-it", "near frontier feel, hungry KV"),
    m!("Qwen3 30B-A3B", 30.5, 3.3, 48, 4, 128, 40960, "qwen3:30b-a3b", "Qwen3-30B-A3B", "MoE: 30B smarts at 3B speed"),
    m!("Qwen3 32B", 32.8, 32.8, 64, 8, 128, 40960, "qwen3:32b", "Qwen3-32B", "the dense heavyweight"),
    m!("Llama 3.3 70B", 70.6, 70.6, 80, 8, 128, 131072, "llama3.3:70b-instruct", "Llama-3.3-70B-Instruct", "the big one"),
];
