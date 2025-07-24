import os
import subprocess
from huggingface_hub import snapshot_download

def download_and_convert_to_gguf(model_repo: str, quant: str, hf_token: str = None, out_dir: str = "converted"):
    """
    Downloads a Hugging Face model and converts it to GGUF format using llama.cpp's convert_hf_to_gguf.py

    Args:
        model_repo (str): e.g. "meta-llama/Llama-2-7b-hf"
        quant (str): "fp16" or "int4" (will use q4_k_m internally)
        hf_token (str, optional): Hugging Face access token
        out_dir (str): directory where gguf file will be saved

    Returns:
        str: path to the generated GGUF file or None if failed
    """
    assert quant in ("fp16", "int4"), "Only fp16 and int4 supported"
    outtype = "f16" if quant == "fp16" else "q4_k_m"

    print(f"📥 Downloading model: {model_repo} ...")
    model_path = snapshot_download(repo_id=model_repo, token=hf_token, local_dir=os.path.join("hf_models", model_repo.replace('/', '_')))

    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir, f"{model_repo.replace('/', '_')}.{outtype}.gguf")

    cmd = [
        "python3", "convert_hf_to_gguf.py",  # llama.cpp's HF conversion script
        "--outfile", output_file,
        "--outtype", outtype,
        model_path
    ]

    print("🚧 Running GGUF conversion...")
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ GGUF conversion complete: {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"❌ GGUF conversion failed: {e}")
        return None
