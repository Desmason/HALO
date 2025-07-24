#!/usr/bin/env python3
"""
hardware_recommend.py

1) Detect local hardware (CPU, RAM, GPU).
2) Recommend the best LLaMA/OPT model (static catalog).
3) Optionally assess any extra HF repo (public or private) via your HF token.
"""

import os
import platform
import psutil
import argparse
from huggingface_hub import HfApi, login
from huggingface_hub.utils import HfHubHTTPError

# ─── Hugging Face Login ────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

# ─── Optional GPU detection ────────────────────────────────────────────────────
try:
    import torch
except ImportError:
    torch = None

# ─── Hardware Detection ────────────────────────────────────────────────────────
def detect_hardware():
    cpu_cores = psutil.cpu_count(logical=True)
    ram_gb = psutil.virtual_memory().total / (1024**3)
    gpu_vram_gb = 0.0
    os_name = platform.system()
    if torch and torch.cuda.is_available():
        gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    elif torch and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        gpu_vram_gb = ram_gb * 0.75  # heuristic for Apple Silicon shared memory
    return {"cpu_cores": cpu_cores, "ram_gb": ram_gb, "gpu_vram_gb": gpu_vram_gb, "os": os_name}

# ─── Model Catalog (Static) ─────────────────────────────────────────────────────
MODEL_CATALOG = [
    {"name": "LLaMA-3.2-1B",  "repo_id": "meta-llama/Llama-3-2-1b"},
    {"name": "LLaMA-3.2-3B",  "repo_id": "meta-llama/Llama-3-2-3b"},
    {"name": "LLaMA-3.1-8B",  "repo_id": "meta-llama/Llama-3-1-8b"},
    {"name": "LLaMA-3.1-70B", "repo_id": "meta-llama/Llama-3-1-70b"},
    {"name": "LLaMA-3.1-405B","repo_id": "meta-llama/Llama-3-1-405b"},
    {"name": "OPT-125M",      "repo_id": "facebook/opt-125m"},
    {"name": "OPT-350M",      "repo_id": "facebook/opt-350m"},
    {"name": "OPT-1.3B",      "repo_id": "facebook/opt-1.3b"},
    {"name": "OPT-2.7B",      "repo_id": "facebook/opt-2.7b"},
    {"name": "OPT-6.7B",      "repo_id": "facebook/opt-6.7b"},
    {"name": "OPT-13B",       "repo_id": "facebook/opt-13b"},
    {"name": "OPT-30B",       "repo_id": "facebook/opt-30b"},
    {"name": "OPT-66B",       "repo_id": "facebook/opt-66b"},
    {"name": "OPT-175B",      "repo_id": "facebook/opt-175b"},
]

# ─── Hugging Face File Size Estimation ─────────────────────────────────────────
_hf_api = HfApi()

def get_model_bytes(repo_id: str, token: str) -> int | None:
    try:
        files = _hf_api.list_repo_files(repo_id, use_auth_token=token)
    except HfHubHTTPError:
        return None

    total = 0
    for path in files:
        if path.endswith((".bin", ".safetensors", ".gguf")):
            try:
                infos = _hf_api.get_paths_info(repo_id, [path], use_auth_token=token)
                for info in infos:
                    if info.size:
                        total += info.size
            except HfHubHTTPError:
                continue
    return total

# ─── Helper Utils ───────────────────────────────────────────────────────────────
def bytes_to_gb(b: int) -> float:
    return b / (1024**3)

def assess_fit(name: str, bytes_fp16: int, specs: dict, fine_tuning: bool = False):
    gpu = specs["gpu_vram_gb"]
    budget = (gpu if gpu > 0 else specs["ram_gb"]) * 0.8
    if fine_tuning:
        budget *= 0.66
    mode = "QLoRA" if fine_tuning else "inference"

    gb_fp16 = bytes_to_gb(bytes_fp16)
    print(f"🔍 Assessing {name} (~{gb_fp16:.1f} GB FP16) [{mode}]:")
    for label, gb in [("FP16", gb_fp16), ("INT4", gb_fp16 / 4)]:
        status = "fits ✅" if gb <= budget else "too big ❌"
        print(f"  {label:<4s}: {gb:6.2f} GB → {status}")
    print()

# ─── Main Entry ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Hardware & HF model recommender")
    parser.add_argument("repos", nargs="*", help="Extra HF repo IDs to assess")
    args = parser.parse_args()

    specs = detect_hardware()
    print("\n🖥️  Hardware specs detected:")
    print(f"  OS:           {specs['os']}")
    print(f"  CPU cores:    {specs['cpu_cores']}")
    print(f"  RAM (GB):     {specs['ram_gb']:.1f}")
    print(f"  GPU VRAM(GB): {specs['gpu_vram_gb']:.1f}\n")

    # Build and sort catalog by FP16 size
    catalog = []
    for m in MODEL_CATALOG:
        b = get_model_bytes(m["repo_id"], HF_TOKEN)
        if b is not None:
            catalog.append((m["name"], b))
    catalog.sort(key=lambda x: x[1], reverse=True)

    def pick_model(budget: float):
        for name, b in catalog:
            gb_fp16 = bytes_to_gb(b)
            for prec in ("fp16", "int4"):
                req = gb_fp16 / (4 if prec == "int4" else 1)
                if req <= budget:
                    return name, prec, req
        # fallback to smallest
        name, b = catalog[-1]
        return name, "int4", bytes_to_gb(b) / 4

    # Recommend models
    base_budget = (specs["gpu_vram_gb"] if specs["gpu_vram_gb"] > 0 else specs["ram_gb"]) * 0.8
    name, prec, req = pick_model(base_budget)
    print(f"🎯 Recommended for inference:    {name} @ {prec.upper()} (≈{req:.1f} GB)")

    q_budget = base_budget * 0.66
    name_ft, prec_ft, req_ft = pick_model(q_budget)
    print(f"🛠️  Recommended for QLoRA tuning: {name_ft} @ {prec_ft.upper()} (≈{req_ft:.1f} GB)\n")

    # Assess extra Hugging Face models from user
    for repo in args.repos:
        b = get_model_bytes(repo, HF_TOKEN)
        if b is None:
            print(f"⚠️  Skipping {repo} (fetch error or permission denied)\n")
        else:
            assess_fit(repo, b, specs, fine_tuning=False)
            assess_fit(repo, b, specs, fine_tuning=True)

if __name__ == "__main__":
    main()