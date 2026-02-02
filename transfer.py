#!/usr/bin/env python3

import subprocess
from pathlib import Path

# Paths you gave me
BASE_GGUF = Path("/Users/henry/UTD/HALO/converted/meta-llama_Llama-3.2-1B-Instruct.q4_k_m.gguf")
LORA_DIR = Path("/Users/henry/UTD/HALO/finetuned_adapters/meta-llama_Llama-3.2-1B-Instruct/peft_lora/hf_trainer_output/checkpoint-9")

# Where your convert_lora_to_gguf.py lives
SCRIPT = Path("/Users/henry/UTD/HALO/convert_lora_to_gguf.py")

# HF ID for Llama 3.2 1B Instruct
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

# Output adapter GGUF path
OUTFILE = LORA_DIR / "meta-llama_Llama-3.2-1B-Instruct-lora-f16.gguf"

def main():
    cmd = [
        "python",
        str(SCRIPT),
        "--base-model-id", BASE_MODEL_ID,  # load base config from HF hub
        "--outtype", "f16",               # adapter weights in F16 (good default)
        "--outfile", str(OUTFILE),
        str(LORA_DIR),                    # this must contain adapter_config.json + adapter_model.safetensors/bin
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Done!")
    print("LoRA adapter GGUF written to:", OUTFILE)

if __name__ == "__main__":
    main()
