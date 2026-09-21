"""What machine is this.

Everything downstream depends on four numbers from here:
  memory_gb        how much RAM (Apple Silicon: unified, so this is also VRAM)
  bandwidth_gbps   memory bandwidth, the single biggest predictor of decode tok/s
  backend          which runtime family to recommend
  vram_gb          discrete GPU memory, None on Apple Silicon

macOS sources
  chip name   : sysctl -n machdep.cpu.brand_string
  total RAM   : sysctl -n hw.memsize            (bytes)
  GPU cores   : system_profiler SPDisplaysDataType
  bandwidth   : not exposed by the OS, so look it up from the chip name below

Linux sources
  RAM         : psutil.virtual_memory().total
  NVIDIA      : nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  bandwidth   : look up from GPU name, or read from nvidia-smi if you find a field
"""

from __future__ import annotations

from dataclasses import dataclass

# Apple Silicon memory bandwidth, GB/s. Apple publishes these on each chip's spec page.
# Match on the longest key that appears in the brand string, e.g. "Apple M3 Max".
# Some chips ship in two bandwidth bins (M3 Max 300/400, M4 Max 410/546); pick the
# lower one unless you can tell them apart by GPU core count.
APPLE_BANDWIDTH_GBPS: dict[str, float] = {
    "M1": 68.25, "M1 Pro": 200, "M1 Max": 400, "M1 Ultra": 800,
    "M2": 100, "M2 Pro": 200, "M2 Max": 400, "M2 Ultra": 800,
    "M3": 100, "M3 Pro": 150, "M3 Max": 300, "M3 Ultra": 800,
    "M4": 120, "M4 Pro": 273, "M4 Max": 410,
    # TODO: add M5 family from Apple's spec pages
}


@dataclass
class Hardware:
    chip: str
    os: str                 # "darwin" | "linux" | "windows"
    memory_gb: float
    bandwidth_gbps: float
    backend: str            # "mlx" | "llama.cpp-metal" | "llama.cpp-cuda" | "vllm" | "llama.cpp-cpu"
    vram_gb: float | None = None
    gpu_cores: int | None = None


def detect() -> Hardware:
    """Probe the machine and fill a Hardware.

    Backend rule of thumb (see README discussion):
      Apple Silicon                     -> mlx (llama.cpp-metal as fallback)
      NVIDIA, single GPU, single user   -> llama.cpp-cuda
      NVIDIA, many users / batching     -> vllm
      no GPU                            -> llama.cpp-cpu
    """
    raise NotImplementedError
