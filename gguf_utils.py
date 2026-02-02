#!/usr/bin/env python3
from pathlib import Path
from typing import Optional, Tuple

from gguf_download import download_and_convert_to_gguf


def _project_root() -> Path:
    """
    Since gguf_utils.py sits in the HALO root, the project root is simply
    the directory containing this file.
    """
    return Path(__file__).resolve().parent


def ensure_base_gguf_exists(
    model_repo: str,
    quant: str,                        # "f16" or "q4_k_m"
    hf_token: Optional[str] = None,
    out_dir: str = "converted",
    do_cleanup: bool = True,
) -> Tuple[str, int]:
    """
    Ensure a base GGUF file exists under <project_root>/<out_dir>.

    - If the expected GGUF file already exists, return (path, 0).
    - If missing, call download_and_convert_to_gguf(...) to create it,
      then return (path, freed_bytes).

    Args:
        model_repo: HF repo id, e.g. "meta-llama/Llama-3.2-1B-Instruct"
        quant:     "f16" (FP16) or "q4_k_m" (INT4)
        hf_token:  Optional HF token
        out_dir:   Relative to the HALO root (default "converted")
        do_cleanup: Whether gguf_download should clean cache/snapshots

    Returns:
        (gguf_path, freed_bytes)
    """
    project_root = _project_root()
    out_dir_path = (project_root / out_dir).resolve()
    out_dir_path.mkdir(parents=True, exist_ok=True)

    safe_name = model_repo.replace("/", "_")

    # Map your external quant name → filename suffix & gguf_download quant
    if quant == "f16":
        suffix = "f16"
        downloader_quant = "fp16"
    elif quant == "q4_k_m":
        suffix = "q4_k_m"
        downloader_quant = "int4"
    else:
        raise ValueError("quant must be 'f16' or 'q4_k_m'")

    expected_path = out_dir_path / f"{safe_name}.{suffix}.gguf"

    # ─────────────────── Case 1: file already exists ───────────────────
    if expected_path.exists():
        print(f"✅ Found existing GGUF ({quant}), skipping download:")
        print(f"   {expected_path}")
        return str(expected_path), 0

    # ─────────────────── Case 2: download + convert ────────────────────
    print(f"⚠️ GGUF not found for {model_repo} ({quant}).")
    print("   Triggering download_and_convert_to_gguf...")

    result = download_and_convert_to_gguf(
        model_repo=model_repo,
        quant=downloader_quant,         # "fp16" or "int4"
        hf_token=hf_token,
        out_dir=str(out_dir_path),
        do_cleanup=do_cleanup,
    )

    if result is None:
        raise RuntimeError("download_and_convert_to_gguf failed")

    gguf_path, freed_bytes = result
    print(f"✅ New GGUF ready at: {gguf_path}")
    print(f"🧹 Freed bytes during cleanup: {freed_bytes}")

    return gguf_path, freed_bytes
