# pages/3_model_selection.py
from pathlib import Path
import streamlit as st

from gguf_download import download_and_convert_to_gguf
from hf_lora_to_gguf import (
    hf_lora_finetune_to_gguf_adapter,
    HfLoraFinetuneError,
)
from gguf_utils import ensure_base_gguf_exists



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
    value=True,
)

user_model_infer = ""
if not use_auto_infer:
    user_model_infer = st.text_input(
        "Enter a model name (HF repo) or a local .gguf path",
        placeholder="e.g. meta-llama/Llama-3.1-8B-Instruct or /path/to/model.gguf",
    )

precision_infer = st.selectbox("Select Quantization for Inference", ["FP16", "INT4"])

if st.button("🚀 Proceed to Inference"):
    chosen = (
        user_model_infer.strip()
        if not use_auto_infer
        else str(st.session_state.auto_model_infer).strip()
    )
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
            # Map UI precision to the exact quant names expected by ensure_base_gguf_exists()
            quant_arg = "f16" if precision_ft.upper() == "FP16" else "q4_k_m"


            out = download_and_convert_to_gguf(
                model_repo=chosen,
                quant=quant_arg,
                hf_token=hf_token,
                out_dir="converted",
                do_cleanup=True,  # cleanup handled inside the function
            )
            if out:
                out_path, freed_bytes = out
                st.session_state.gguf_path = out_path
                if freed_bytes and freed_bytes > 0:
                    st.info(
                        f"🧹 Cleaned intermediates & cache. Freed {_human_bytes(freed_bytes)}."
                    )
                st.success(f"GGUF model ready at: {out_path}")
                st.switch_page("pages/4_inference.py")
            else:
                st.error("Failed to prepare the GGUF model.")

st.markdown("---")

# ─── Option 2: Fine-Tune with HF LoRA + convert_lora_to_gguf ─────────────────────────
st.subheader("Option 2: Fine-Tune Then Use (HF LoRA → GGUF adapter)")

use_auto_ft = st.checkbox(
    f"Use Auto-Recommended Fine-Tune Model ({st.session_state.auto_model_ft})",
    value=True,
)

user_model_ft = ""
if not use_auto_ft:
    user_model_ft = st.text_input(
        "Enter a base model (HF repo) for fine-tuning",
        placeholder="e.g. meta-llama/Llama-3.1-8B-Instruct",
    )

precision_ft = st.selectbox(
    "Select Quantization for Inference (base GGUF)", ["FP16", "INT4"], key="ft_precision"
)

if st.button("🧪 Start Fine-Tuning"):
    # This is the **HF repo id** used for LoRA training
    base_model_repo = (
        user_model_ft.strip()
        if not use_auto_ft
        else str(st.session_state.auto_model_ft).strip()
    )
    if not base_model_repo:
        st.error("No base HF model specified for fine-tuning.")
        st.stop()

    # For inference later, we still need a quantized GGUF base model
    quant_arg = "f16" if precision_ft.upper() == "FP16" else "q4_k_m"

    # Get training data path from Step 2
    train_data_path = st.session_state.get("train_data_path")
    if not train_data_path or not Path(train_data_path).exists():
        st.error("No training data found. Please upload .txt files in Step 2.")
        st.stop()

    llama_cpp_dir = st.session_state.get("llama_cpp_dir", "llama.cpp")
    hf_token = st.session_state.get("hf_token", None)

    # 1) Make sure we have a base GGUF ready for inference
    st.subheader("🧱 Preparing Base GGUF for Inference")
    with st.spinner("Checking / preparing quantized base GGUF…"):
        try:
            base_gguf_path, freed_bytes = ensure_base_gguf_exists(
                model_repo=base_model_repo,
                quant=quant_arg,
                hf_token=hf_token,
                out_dir="converted",
                do_cleanup=True,
            )
        except Exception as e:
            st.error(
                "Failed to prepare the base GGUF model for inference.\n\n"
                f"Error: {e}"
            )
            st.stop()

        if freed_bytes and freed_bytes > 0:
            st.info(f"🧹 Cleaned intermediates & cache. Freed {_human_bytes(freed_bytes)}.")

        st.success(f"Base GGUF for inference ready at: {base_gguf_path}")


    # 2) HF LoRA fine-tuning → PEFT adapter → convert_lora_to_gguf.py
    st.subheader("📡 Fine-Tuning Progress (HF LoRA)")
    progress_box = st.empty()

    with st.spinner("Running HF LoRA fine-tuning and converting to GGUF adapter…"):
        try:
            # NOTE: this trains in HF/PEFT, then calls convert_lora_to_gguf.py
            adapter_gguf_path = hf_lora_finetune_to_gguf_adapter(
                hf_model_id=base_model_repo,
                train_txt_path=train_data_path,
                llama_cpp_dir=llama_cpp_dir,
                hf_token=hf_token,
                out_dir="finetuned_adapters",
                num_epochs=1,        # you can expose this in UI later
                batch_size=1,
                grad_accum_steps=4,
                learning_rate=1e-4,
                use_4bit=False,      # set True if you have bitsandbytes + GPU
                outtype="f16",
            )
        except HfLoraFinetuneError as e:
            st.error(f"LoRA fine-tuning failed:\n{e}")
            st.stop()

    progress_box.success("HF LoRA fine-tuning + GGUF adapter conversion completed.")

    # Save info for inference step:
    # - base_gguf_path: quantized base model
    # - lora_adapter_gguf: GGUF LoRA adapter produced by convert_lora_to_gguf.py
    st.session_state.model_name = base_model_repo
    st.session_state.precision_choice = precision_ft
    st.session_state.base_gguf_path = base_gguf_path
    st.session_state.lora_adapter_gguf = str(adapter_gguf_path)

    st.success(
        "✅ LoRA GGUF adapter ready at: "
        f"`{adapter_gguf_path}`\n\n"
        "You can now run inference with the base GGUF + this adapter."
    )
    st.switch_page("pages/4_inference.py")


# ─── Back Navigation ───────────────────────────
if st.button("⬅️ Back to Setup"):
    st.switch_page("pages/2_setup_recommendation.py")
