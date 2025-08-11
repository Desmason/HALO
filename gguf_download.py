import os
import subprocess
from huggingface_hub import snapshot_download
from pathlib import Path

def download_and_convert_to_gguf(model_repo: str, quant: str, hf_token: str = None, out_dir: str = "converted"):
    """
    Downloads a Hugging Face model, converts to GGUF (FP16), and if requested,
    quantizes to INT4 (q4_k_m) using llama.cpp's llama-quantize tool.
    """
    assert quant in ("fp16", "int4"), "Only fp16 and int4 supported"
    is_int4 = (quant == "int4")

    # Always start from FP16
    outtype = "f16"

    # Download model snapshot from Hugging Face
    print(f"📥 Downloading model from Hugging Face: {model_repo}")
    model_dir = snapshot_download(
        repo_id=model_repo,
        token=hf_token,
        allow_patterns=["*.bin", "*.safetensors", "*.json"],
        local_dir=os.path.join("hf_models", model_repo.replace('/', '_')),
        local_dir_use_symlinks=False
    )

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    fp16_path = os.path.join(out_dir, f"{model_repo.replace('/', '_')}.f16.gguf")

    # Step 1: Convert to GGUF FP16
    cmd_convert = [
        "python3", "convert_hf_to_gguf.py",
        "--outfile", fp16_path,
        "--outtype", outtype,
        model_dir
    ]
    print("🚧 Running GGUF conversion to FP16...")
    print(" ".join(cmd_convert))
    try:
        subprocess.run(cmd_convert, check=True)
    except subprocess.CalledProcessError as e:
        print("❌ GGUF conversion failed")
        return None

    # Step 2: Quantize to INT4 if requested
    if is_int4:
        int4_path = os.path.join(out_dir, f"{model_repo.replace('/', '_')}.q4_k_m.gguf")
        cmd_quant = [
            "./llama.cpp/build/bin/llama-quantize",
            fp16_path,
            int4_path,
            "q4_k_m"
        ]
        print("🚧 Running llama-quantize to INT4 (q4_k_m)...")
        print(" ".join(cmd_quant))
        try:
            subprocess.run(cmd_quant, check=True)
            print(f"✅ Quantized model ready at: {int4_path}")
            return int4_path
        except subprocess.CalledProcessError:
            print("❌ Quantization failed")
            return None

    print(f"✅ FP16 GGUF model ready at: {fp16_path}")
    return fp16_path
