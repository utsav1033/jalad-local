"""The contestants. Architecture numbers from each model's config.json."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    name: str
    params_b: float        # total parameters, billions (memory)
    active_b: float        # parameters touched per token (speed); equals params_b unless MoE
    layers: int
    kv_heads: int
    head_dim: int
    max_context: int
    ollama: str            # base tag, quant suffix appended
    mlx: str               # mlx-community repo stem, "-4bit" etc appended
    tagline: str

    @property
    def is_moe(self) -> bool:
        return self.active_b < self.params_b


CATALOG: list[Model] = [
    Model("Llama 3.2 1B", 1.24, 1.24, 16, 8, 64, 131072,
          "llama3.2:1b-instruct", "Llama-3.2-1B-Instruct", "tiny, for autocomplete and toys"),
    Model("Llama 3.2 3B", 3.21, 3.21, 28, 8, 128, 131072,
          "llama3.2:3b-instruct", "Llama-3.2-3B-Instruct", "small but coherent"),
    Model("Gemma 3 4B", 4.3, 4.3, 34, 4, 256, 131072,
          "gemma3:4b-it", "gemma-3-4b-it", "small, sees images"),
    Model("Qwen2.5 7B", 7.6, 7.6, 28, 4, 128, 32768,
          "qwen2.5:7b-instruct", "Qwen2.5-7B-Instruct", "the reliable 7B"),
    Model("Llama 3.1 8B", 8.03, 8.03, 32, 8, 128, 131072,
          "llama3.1:8b-instruct", "Meta-Llama-3.1-8B-Instruct", "the default everyone benchmarks"),
    Model("Qwen3 8B", 8.2, 8.2, 36, 8, 128, 40960,
          "qwen3:8b", "Qwen3-8B", "thinking mode, the new default 8B"),
    Model("Gemma 3 12B", 12.2, 12.2, 48, 8, 256, 131072,
          "gemma3:12b-it", "gemma-3-12b-it", "strong writer, sees images"),
    Model("Qwen3 14B", 14.8, 14.8, 40, 8, 128, 40960,
          "qwen3:14b", "Qwen3-14B", "thinking mode, good at code"),
    Model("Phi-4 14B", 14.7, 14.7, 40, 10, 128, 16384,
          "phi4:14b", "phi-4", "punches above its weight on reasoning"),
    Model("Mistral Small 3.1 24B", 24.0, 24.0, 40, 8, 128, 131072,
          "mistral-small3.1:24b-instruct-2503", "Mistral-Small-3.1-24B-Instruct-2503", "the sweet spot if it fits"),
    Model("Gemma 3 27B", 27.4, 27.4, 62, 16, 128, 131072,
          "gemma3:27b-it", "gemma-3-27b-it", "near frontier feel, hungry KV"),
    Model("Qwen3 30B-A3B", 30.5, 3.3, 48, 4, 128, 40960,
          "qwen3:30b-a3b", "Qwen3-30B-A3B", "MoE: 30B smarts at 3B speed"),
    Model("Qwen3 32B", 32.8, 32.8, 64, 8, 128, 40960,
          "qwen3:32b", "Qwen3-32B", "the dense heavyweight"),
    Model("Llama 3.3 70B", 70.6, 70.6, 80, 8, 128, 131072,
          "llama3.3:70b-instruct", "Llama-3.3-70B-Instruct", "the big one"),
]
