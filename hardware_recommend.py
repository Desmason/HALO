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

# at the bottom of hardware_recommend.py

if __name__ == "__main__":
    specs = detect_hardware()
    print("🖥️  Hardware specs detected:")
    for k, v in specs.items():
        print(f"  {k}: {v}")

