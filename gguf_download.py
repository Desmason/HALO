# gguf_download.py
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple
from huggingface_hub import snapshot_download

# ───────────────────────── helpers ─────────────────────────

def _find_convert_script() -> Optional[str]:
    candidates = [
        Path("convert_hf_to_gguf.py"),
        Path.cwd() / "convert_hf_to_gguf.py",
        Path(__file__).parent / "convert_hf_to_gguf.py",
        Path("/mnt/data/convert_hf_to_gguf.py"),   # optional common location
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    return "convert_hf_to_gguf.py"  # hope it's importable

def _find_llama_quantize() -> Optional[str]:
    env_bin = os.environ.get("LLAMA_QUANT_BIN")
    if env_bin and Path(env_bin).exists():
        return env_bin

    common = [
        Path("llama.cpp") / "build" / "bin" / "llama-quantize",
        Path("build") / "bin" / "llama-quantize",
        Path("llama-quantize"),
    ]
    for c in common:
        if c.exists():
            return str(c.resolve())

    which = shutil.which("llama-quantize")
    return which if which else None

def _looks_like_hf_repo_id(s: str) -> bool:
    return ("/" in s) and (not Path(s).expanduser().exists())

def _hf_cache_root() -> Path:
    env_home = os.environ.get("HF_HOME")
    if env_home:
        return Path(env_home).expanduser().resolve()
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "huggingface"
    return Path.home() / ".cache" / "huggingface"

def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except Exception:
                pass
    return total

def _rmtree_size(path: Path) -> int:
    if not path.exists():
        return 0
    size = _dir_size(path)
    shutil.rmtree(path, ignore_errors=True)
    return size

def _cleanup_repo_artifacts(final_gguf: Path, repo_id: str, hf_models_root: str, out_dir: Path) -> int:
    """
    Keep ONLY the final .gguf for this repo. Delete:
      1) hf_models/<org_repo> snapshot dir
      2) HF cache for this repo (models--org--repo)
      3) Any artifacts in converted/ that start with the repo prefix,
         EXCEPT the final .gguf itself.
    Returns bytes freed.
    """
    freed = 0
    repo_prefix = repo_id.replace("/", "_")

    # 1) local HF snapshot
    snapshot_dir = Path(hf_models_root) / repo_prefix
    if snapshot_dir.exists():
        freed += _rmtree_size(snapshot_dir)
        try:
            Path(hf_models_root).rmdir()
        except OSError:
            pass

    # 2) HF cache
    if _looks_like_hf_repo_id(repo_id):
        safe_repo = repo_id.replace("/", "--")
        hf_repo_dir = _hf_cache_root() / "hub" / f"models--{safe_repo}"
        if hf_repo_dir.exists():
            freed += _rmtree_size(hf_repo_dir)

    # 3) converted/ repo artifacts except final .gguf
    for p in out_dir.iterdir():
        try:
            if p.resolve() == final_gguf.resolve():
                continue
            if not p.name.startswith(repo_prefix):
                continue

            if p.is_file():
                try:
                    size = p.stat().st_size
                except Exception:
                    size = 0
                p.unlink(missing_ok=True)
                freed += size
            elif p.is_dir():
                freed += _rmtree_size(p)
        except Exception:
            pass

    return freed

# ───────────────────────── main API ─────────────────────────

def download_and_convert_to_gguf(
    model_repo: str,
    quant: str,                               # "fp16" or "int4"
    hf_token: Optional[str] = None,
    out_dir: str = "converted",
    do_cleanup: bool = True,                  # ← cleanup handled inside
) -> Optional[Tuple[str, int]]:
    """
    Download HF model, convert to GGUF FP16, optionally quantize to INT4 (q4_k_m),
    then (optionally) cleanup snapshots/cache/intermediates.

    Returns:
        tuple[str, int] | None:
            (final_gguf_path, freed_bytes) or None on failure.
    """
    assert quant in ("fp16", "int4"), "Only fp16 and int4 supported"
    is_int4 = (quant == "int4")

    # Download HF snapshot (only necessary files)
    repo_prefix = model_repo.replace("/", "_")
    hf_models_root = "hf_models"
    local_dir = Path(hf_models_root) / repo_prefix
    local_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"📥 Downloading model from Hugging Face: {model_repo}")
    model_dir = snapshot_download(
        repo_id=model_repo,
        token=hf_token,
        allow_patterns=[
            "*.bin",
            "*.safetensors",
            "*.json",
            "*.model",
            "tokenizer.json",
        ],
        local_dir=str(local_dir),
        local_dir_use_symlinks=False
    )

    out_dir_p = Path(out_dir).expanduser().resolve()
    out_dir_p.mkdir(parents=True, exist_ok=True)

    # Convert to FP16 GGUF
    fp16_path = out_dir_p / f"{repo_prefix}.f16.gguf"
    convert_script = _find_convert_script()
    cmd_convert = [
        "python3", convert_script,
        "--outfile", str(fp16_path),
        "--outtype", "f16",
        str(model_dir),
    ]
    print("🚧 Converting to GGUF (FP16)…")
    print(" ".join(cmd_convert))
    try:
        subprocess.run(cmd_convert, check=True)
    except subprocess.CalledProcessError:
        print("❌ GGUF conversion failed")
        return None

    final_path = fp16_path

    # Quantize to INT4 if requested
    if is_int4:
        int4_path = out_dir_p / f"{repo_prefix}.q4_k_m.gguf"
        quant_bin = _find_llama_quantize()
        if not quant_bin:
            print("❌ Could not find `llama-quantize`. Build llama.cpp or set LLAMA_QUANT_BIN.")
            return None

        cmd_quant = [quant_bin, str(fp16_path), str(int4_path), "q4_k_m"]
        print("🚧 Running llama-quantize (INT4 q4_k_m)…")
        print(" ".join(cmd_quant))
        try:
            subprocess.run(cmd_quant, check=True)
        except subprocess.CalledProcessError:
            print("❌ Quantization failed")
            return None

        # INT4 chosen → delete FP16 intermediate immediately
        try:
            if fp16_path.exists():
                fp16_path.unlink()
                print(f"🧹 Deleted FP16 intermediate: {fp16_path}")
        except Exception as e:
            print(f"⚠️ Could not delete FP16 intermediate: {e}")

        final_path = int4_path

    freed_bytes = 0
    if do_cleanup:
        freed_bytes = _cleanup_repo_artifacts(
            final_gguf=final_path,
            repo_id=model_repo,
            hf_models_root=hf_models_root,
            out_dir=out_dir_p,
        )
        if freed_bytes > 0:
            print(f"🧹 Cleanup complete. Freed ~{freed_bytes} bytes.")

    print(f"✅ Model ready at: {final_path}")
    return (str(final_path.resolve()), int(freed_bytes))
