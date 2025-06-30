import os
import psutil
import platform
import subprocess
import sys

# Try importing torch for GPU detection; it's optional if you just want CPU-only.
try:
    import torch
except ImportError:
    torch = None


# ─── Hardware Detection ────────────────────────────────────────────────────────

def detect_hardware():
    """Detect GPU VRAM (GB), total system RAM (GB), and CPU core count."""
    # CPU
    cpu_cores = psutil.cpu_count(logical=True)
    # RAM
    ram_gb = psutil.virtual_memory().total / (1024**3)

    # GPU VRAM
    gpu_vram_gb = 0.0
    conf = platform.system()

    if torch and torch.cuda.is_available():
        # NVIDIA CUDA
        prop = torch.cuda.get_device_properties(0)
        gpu_vram_gb = prop.total_memory / (1024**3)
    elif torch and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        # Apple MPS: treat unified memory as GPU
        # We assume ~75% of total RAM is usable for the model.
        gpu_vram_gb = ram_gb * 0.75
    else:
        # No discrete GPU detected → CPU-only
        gpu_vram_gb = 0.0

    return {
        "cpu_cores": cpu_cores,
        "ram_gb": ram_gb,
        "gpu_vram_gb": gpu_vram_gb,
        "os": conf,
    }

# ─── Model Catalog & Helpers ───────────────────────────────────────────────────

# List of candidate models with their parameter counts
MODEL_CATALOG = [

    # LLaMA-2 family
    {"name": "LLaMA-2-7B",   "params_b":   7e9},
    {"name": "LLaMA-2-13B",  "params_b":  13e9},
    {"name": "LLaMA-2-70B",  "params_b":  70e9},

    # OPT family
    {"name": "OPT-125M",     "params_b": 0.125e9},
    {"name": "OPT-350M",     "params_b": 0.350e9},
    {"name": "OPT-1.3B",     "params_b":   1.3e9},
    {"name": "OPT-2.7B",     "params_b":   2.7e9},
    {"name": "OPT-6.7B",     "params_b":   6.7e9},
    {"name": "OPT-13B",      "params_b":  13e9},
    {"name": "OPT-30B",      "params_b":  30e9},
    {"name": "OPT-66B",      "params_b":  66e9},
    {"name": "OPT-175B",     "params_b": 175e9},
]


BYTES_PER_PARAM = {
    "fp16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}


# at the bottom of hardware_recommend.py

if __name__ == "__main__":
    specs = detect_hardware()
    print("🖥️  Hardware specs detected:")
    for k, v in specs.items():
        print(f"  {k}: {v}")

