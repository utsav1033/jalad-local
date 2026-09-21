"""The contestants.

Keep this small and hand-checked. Ten families is plenty for v1.
Architecture numbers come from each model's config.json on Hugging Face:
  num_hidden_layers, hidden_size, num_key_value_heads, head_dim (or hidden_size / num_attention_heads)

The KV cache formula in budget.py needs layers, kv_heads and head_dim, not hidden_size.
Models with GQA (most modern ones) have far fewer kv_heads than attention heads,
which is why an 8B Llama 3 has a much smaller KV cache than an 7B Llama 2.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    name: str
    params_b: float        # billions of parameters
    layers: int
    kv_heads: int
    head_dim: int
    max_context: int       # what the model was trained for
    hf_repo: str           # base repo, used to build the run command later


# Seeded with two so the shape is clear. Verify these against config.json, then add the rest.
CATALOG: list[Model] = [
    Model("Llama 3.1 8B", 8.0, 32, 8, 128, 131072, "meta-llama/Llama-3.1-8B-Instruct"),
    Model("Qwen2.5 7B", 7.6, 28, 4, 128, 32768, "Qwen/Qwen2.5-7B-Instruct"),
    # TODO: Qwen3, Gemma 3, Mistral Small, Phi-4, DeepSeek-R1 distills, a 1-3B class, a 30B+ class, a 70B class
]
