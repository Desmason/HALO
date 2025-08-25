# pages/3_model_selection.py
from pathlib import Path
import streamlit as st
from gguf_download import download_and_convert_to_gguf

def _is_local_gguf(s: str) -> bool:
    try:
        p = Path(s).expanduser().resolve()
        return p.exists() and p.is_file() and p.suffix.lower() == ".gguf"
    except Exception:
        return False

def _human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if n < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} EB"

# ─────────────────────────── Streamlit UI ───────────────────────────
st.set_page_config(page_title="Step 3 – Model Selection", layout="centered")
st.title("📦 Step 3: Choose Model Usage Mode")

if "auto_model_infer" not in st.session_state:
    st.warning("Auto-recommendation missing. Please return to Step 2.")
    st.stop()

# Fallback for fine-tune model if missing
if "auto_model_ft" not in st.session_state or not st.session_state["auto_model_ft"]:
    st.session_state["auto_model_ft"] = st.session_state["auto_model_infer"]

st.markdown("Choose how you'd like to proceed with the model you selected or uploaded.")

# ─── Option 1: Direct Inference ───────────────────────
st.subheader("Option 1: Direct Inference (no fine-tuning)")
use_auto_infer = st.checkbox(
    f"Use Auto-Recommended Model ({st.session_state.auto_model_infer})",
    value=True
)

user_model_infer = ""
if not use_auto_infer:
    user_model_infer = st.text_input(
        "Enter a model name (HF repo) or a local .gguf path",
        placeholder="e.g. meta-llama/Llama-3.1-8B-Instruct or /path/to/model.gguf"
    )

precision_infer = st.selectbox("Select Quantization for Inference", ["FP16", "INT4"])

if st.button("🚀 Proceed to Inference"):
    chosen = user_model_infer.strip() if not use_auto_infer else str(st.session_state.auto_model_infer).strip()
    if not chosen:
        st.error("No model specified.")
        st.stop()

    st.session_state.model_name = chosen
    st.session_state.precision_choice = precision_infer
    st.success(f"Model `{chosen}` selected with {precision_infer}.")

    # If user provided a ready-made .gguf, just use it.
    if _is_local_gguf(chosen):
        st.session_state.gguf_path = str(Path(chosen).expanduser().resolve())
        st.success(f"Using local GGUF: {st.session_state.gguf_path}")
        st.switch_page("pages/4_inference.py")
    else:
        with st.spinner("Preparing model (download → convert → quantize → cleanup)…"):
            hf_token = st.session_state.get("hf_token", None)
            quant_arg = "fp16" if precision_infer.upper() == "FP16" else "int4"

            out = download_and_convert_to_gguf(
                model_repo=chosen,
                quant=quant_arg,
                hf_token=hf_token,
                out_dir="converted",
                do_cleanup=True,              # cleanup handled inside the function
            )
            if out:
                out_path, freed_bytes = out
                st.session_state.gguf_path = out_path
                if freed_bytes and freed_bytes > 0:
                    st.info(f"🧹 Cleaned intermediates & cache. Freed {_human_bytes(freed_bytes)}.")
                st.success(f"GGUF model ready at: {out_path}")
                st.switch_page("pages/4_inference.py")
            else:
                st.error("Failed to prepare the GGUF model.")

st.markdown("---")

# ─── Option 2: Fine-Tune First (placeholder) ─────────────────────────
st.subheader("Option 2: Fine-Tune Then Use")
use_auto_ft = st.checkbox(
    f"Use Auto-Recommended Fine-Tune Model ({st.session_state.auto_model_ft})",
    value=True
)

user_model_ft = ""
if not use_auto_ft:
    user_model_ft = st.text_input(
        "Enter a base model (HF repo) for fine-tuning",
        placeholder="e.g. meta-llama/Llama-3.1-8B"
    )

precision_ft = st.selectbox("Select Quantization for Fine-Tuning", ["FP16", "INT4"], key="ft_precision")

if st.button("🧪 Start Fine-Tuning"):
    st.session_state.model_name = user_model_ft.strip() if not use_auto_ft else st.session_state.auto_model_ft
    st.session_state.precision_choice = precision_ft
    st.success(f"Started fine-tuning `{st.session_state.model_name}` with {precision_ft}. (placeholder)")
    st.switch_page("pages/4_inference.py")

# ─── Back Navigation ───────────────────────────────────
if st.button("⬅️ Back to Setup"):
    st.switch_page("pages/2_setup_recommendation.py")
