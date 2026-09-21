"""What machine is this: chip, memory, bandwidth, backend."""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
from dataclasses import dataclass

import psutil

GIB = 1024**3

# Memory bandwidth in GB/s from Apple's spec pages. Longest matching key wins.
# Chips that ship in two bins (M3 Max 300/400, M4 Max 410/546) use the lower bin.
APPLE_BANDWIDTH_GBPS: dict[str, float] = {
    "M1": 68.25, "M1 Pro": 200, "M1 Max": 400, "M1 Ultra": 800,
    "M2": 100, "M2 Pro": 200, "M2 Max": 400, "M2 Ultra": 800,
    "M3": 100, "M3 Pro": 150, "M3 Max": 300, "M3 Ultra": 800,
    "M4": 120, "M4 Pro": 273, "M4 Max": 410,
    "M5": 153, "M5 Pro": 300, "M5 Max": 450,
}

NVIDIA_BANDWIDTH_GBPS: dict[str, float] = {
    "H100": 3350, "A100": 1935, "L40S": 864, "A10": 600, "T4": 320,
    "RTX 6000 Ada": 960, "RTX A6000": 768,
    "RTX 5090": 1792, "RTX 5080": 960, "RTX 5070 Ti": 896, "RTX 5070": 672,
    "RTX 4090": 1008, "RTX 4080": 717, "RTX 4070 Ti": 504, "RTX 4070": 504, "RTX 4060 Ti": 288, "RTX 4060": 272,
    "RTX 3090": 936, "RTX 3080": 760, "RTX 3070": 448, "RTX 3060": 360,
}

# Apple caps how much unified memory the GPU may wire by default (roughly 75%).
APPLE_GPU_SHARE = 0.75
# Discrete GPUs: keep a little VRAM for the display and the runtime.
CUDA_RESERVE_GIB = 1.0
# CPU-only: leave room for the OS.
CPU_RESERVE_GIB = 4.0
# Fallback when no lookup matches: dual-channel DDR5 territory.
UNKNOWN_BANDWIDTH_GBPS = 60.0


@dataclass
class Hardware:
    chip: str
    os: str
    memory_gib: float
    usable_gib: float           # what a model may actually occupy
    bandwidth_gbps: float
    bandwidth_estimated: bool
    backend: str                # "mlx" | "llama.cpp-cuda" | "llama.cpp-cpu"
    vram_gib: float | None = None
    gpu_name: str | None = None
    gpu_cores: int | None = None

    @property
    def is_apple(self) -> bool:
        return self.backend == "mlx"

    @property
    def memory_label(self) -> str:
        if self.vram_gib is not None:
            return f"{self.vram_gib:.0f} GiB VRAM  (+{self.memory_gib:.0f} GiB RAM)"
        if self.is_apple:
            return f"{self.memory_gib:.0f} GiB unified"
        return f"{self.memory_gib:.0f} GiB RAM"


def _sysctl(key: str) -> str | None:
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _lookup(table: dict[str, float], name: str) -> float | None:
    for key in sorted(table, key=len, reverse=True):
        if re.search(rf"\b{re.escape(key)}\b", name):
            return table[key]
    return None


def _apple_gpu_cores() -> int | None:
    try:
        out = subprocess.check_output(["system_profiler", "SPDisplaysDataType"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    m = re.search(r"Total Number of Cores:\s*(\d+)", out)
    return int(m.group(1)) if m else None


def _nvidia() -> tuple[str, float] | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    first = out.strip().splitlines()[0]
    name, mib = [x.strip() for x in first.split(",")]
    return name, float(mib) / 1024


def detect() -> Hardware:
    system = platform.system().lower()
    memory_gib = psutil.virtual_memory().total / GIB

    if system == "darwin":
        chip = _sysctl("machdep.cpu.brand_string") or platform.processor() or "unknown"
        if "Apple" in chip:
            bw = _lookup(APPLE_BANDWIDTH_GBPS, chip)
            return Hardware(
                chip=chip, os=system, memory_gib=memory_gib,
                usable_gib=memory_gib * APPLE_GPU_SHARE,
                bandwidth_gbps=bw or UNKNOWN_BANDWIDTH_GBPS, bandwidth_estimated=bw is None,
                backend="mlx", gpu_cores=_apple_gpu_cores(),
            )
        return Hardware(
            chip=chip, os=system, memory_gib=memory_gib,
            usable_gib=max(memory_gib - CPU_RESERVE_GIB, 1),
            bandwidth_gbps=UNKNOWN_BANDWIDTH_GBPS, bandwidth_estimated=True,
            backend="llama.cpp-cpu",
        )

    chip = platform.processor() or platform.machine() or "unknown"
    gpu = _nvidia()
    if gpu:
        name, vram = gpu
        bw = _lookup(NVIDIA_BANDWIDTH_GBPS, name)
        return Hardware(
            chip=chip, os=system, memory_gib=memory_gib,
            usable_gib=max(vram - CUDA_RESERVE_GIB, 1),
            bandwidth_gbps=bw or 400.0, bandwidth_estimated=bw is None,
            backend="llama.cpp-cuda", vram_gib=vram, gpu_name=name,
        )
    return Hardware(
        chip=chip, os=system, memory_gib=memory_gib,
        usable_gib=max(memory_gib - CPU_RESERVE_GIB, 1),
        bandwidth_gbps=UNKNOWN_BANDWIDTH_GBPS, bandwidth_estimated=True,
        backend="llama.cpp-cpu",
    )
