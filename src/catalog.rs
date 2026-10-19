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
    /// Hugging Face repo holding GGUF quants, one file per quant.
    pub gguf: &'static str,
    pub tagline: &'static str,
}

impl Model {
    pub fn is_moe(&self) -> bool {
        self.active_b < self.params_b
    }
}

macro_rules! m {
    ($name:expr, $p:expr, $a:expr, $l:expr, $kv:expr, $hd:expr, $ctx:expr, $ollama:expr, $mlx:expr, $gguf:expr, $tag:expr) => {
        Model { name: $name, params_b: $p, active_b: $a, layers: $l, kv_heads: $kv, head_dim: $hd,
                max_context: $ctx, ollama: $ollama, mlx: $mlx, gguf: $gguf, tagline: $tag }
    };
}

pub const CATALOG: &[Model] = &[
    m!("Gemma 3 1B", 1.0, 1.0, 26, 1, 256, 32768, "gemma3:1b-it", "gemma-3-1b-it", "bartowski/google_gemma-3-1b-it-GGUF", "pocket sized, surprisingly chatty"),
    m!("Llama 3.2 1B", 1.24, 1.24, 16, 8, 64, 131072, "llama3.2:1b-instruct", "Llama-3.2-1B-Instruct", "bartowski/Llama-3.2-1B-Instruct-GGUF", "tiny, for autocomplete and toys"),
    m!("Qwen3 1.7B", 2.03, 2.03, 28, 8, 128, 40960, "qwen3:1.7b", "Qwen3-1.7B", "bartowski/Qwen_Qwen3-1.7B-GGUF", "tiny thinker"),
    m!("Llama 3.2 3B", 3.21, 3.21, 28, 8, 128, 131072, "llama3.2:3b-instruct", "Llama-3.2-3B-Instruct", "bartowski/Llama-3.2-3B-Instruct-GGUF", "small but coherent"),
    m!("Phi-4 mini 3.8B", 3.84, 3.84, 32, 8, 128, 131072, "phi4-mini:3.8b", "Phi-4-mini-instruct", "bartowski/microsoft_Phi-4-mini-instruct-GGUF", "small, good at maths and logic"),
    m!("Qwen3 4B", 4.02, 4.02, 36, 8, 128, 40960, "qwen3:4b", "Qwen3-4B", "bartowski/Qwen_Qwen3-4B-GGUF", "best brain per gigabyte"),
    m!("Gemma 3 4B", 4.3, 4.3, 34, 4, 256, 131072, "gemma3:4b-it", "gemma-3-4b-it", "bartowski/google_gemma-3-4b-it-GGUF", "small, sees images"),
    m!("Qwen2.5 7B", 7.6, 7.6, 28, 4, 128, 32768, "qwen2.5:7b-instruct", "Qwen2.5-7B-Instruct", "bartowski/Qwen2.5-7B-Instruct-GGUF", "the reliable 7B"),
    m!("Qwen2.5 Coder 7B", 7.6, 7.6, 28, 4, 128, 32768, "qwen2.5-coder:7b-instruct", "Qwen2.5-Coder-7B-Instruct", "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF", "code completion and edits"),
    m!("DeepSeek R1 Distill 7B", 7.6, 7.6, 28, 4, 128, 131072, "deepseek-r1:7b-qwen-distill", "DeepSeek-R1-Distill-Qwen-7B", "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF", "thinks out loud before answering"),
    m!("Llama 3.1 8B", 8.03, 8.03, 32, 8, 128, 131072, "llama3.1:8b-instruct", "Meta-Llama-3.1-8B-Instruct", "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "the default everyone benchmarks"),
    m!("Qwen3 8B", 8.2, 8.2, 36, 8, 128, 40960, "qwen3:8b", "Qwen3-8B", "bartowski/Qwen_Qwen3-8B-GGUF", "thinking mode, the new default 8B"),
    m!("Gemma 3 12B", 12.2, 12.2, 48, 8, 256, 131072, "gemma3:12b-it", "gemma-3-12b-it", "bartowski/google_gemma-3-12b-it-GGUF", "strong writer, sees images"),
    m!("Qwen3 14B", 14.8, 14.8, 40, 8, 128, 40960, "qwen3:14b", "Qwen3-14B", "bartowski/Qwen_Qwen3-14B-GGUF", "thinking mode, good at code"),
    m!("Qwen2.5 Coder 14B", 14.7, 14.7, 48, 8, 128, 32768, "qwen2.5-coder:14b-instruct", "Qwen2.5-Coder-14B-Instruct", "bartowski/Qwen2.5-Coder-14B-Instruct-GGUF", "serious local coding model"),
    m!("Phi-4 14B", 14.7, 14.7, 40, 10, 128, 16384, "phi4:14b", "phi-4", "bartowski/phi-4-GGUF", "punches above its weight on reasoning"),
    m!("Mistral Small 3.1 24B", 24.0, 24.0, 40, 8, 128, 131072, "mistral-small3.1:24b-instruct-2503", "Mistral-Small-3.1-24B-Instruct-2503", "bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF", "the sweet spot if it fits"),
    m!("Gemma 3 27B", 27.4, 27.4, 62, 16, 128, 131072, "gemma3:27b-it", "gemma-3-27b-it", "bartowski/google_gemma-3-27b-it-GGUF", "near frontier feel, hungry KV"),
    m!("Qwen3 30B-A3B", 30.5, 3.3, 48, 4, 128, 40960, "qwen3:30b-a3b", "Qwen3-30B-A3B", "bartowski/Qwen_Qwen3-30B-A3B-GGUF", "MoE: 30B smarts at 3B speed"),
    m!("Qwen3 32B", 32.8, 32.8, 64, 8, 128, 40960, "qwen3:32b", "Qwen3-32B", "bartowski/Qwen_Qwen3-32B-GGUF", "the dense heavyweight"),
    m!("Llama 3.3 70B", 70.6, 70.6, 80, 8, 128, 131072, "llama3.3:70b-instruct", "Llama-3.3-70B-Instruct", "bartowski/Llama-3.3-70B-Instruct-GGUF", "the big one"),
];
