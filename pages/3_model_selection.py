import streamlit as st
from gguf_download import download_and_convert_to_gguf

st.set_page_config(page_title="Step 3 – Model Selection", layout="centered")
st.title("📦 Step 3: Choose Model Usage Mode")

if "auto_model_infer" not in st.session_state:
    st.warning("Auto-recommendation missing. Please return to Step 2.")
    st.stop()

st.markdown("Choose how you'd like to proceed with the model you selected or uploaded.")

# ─── Option 1: Direct Inference ───────────────────────
st.subheader("Option 1: Direct Inference (no fine-tuning)")
use_auto_infer = st.checkbox(f"Use Auto-Recommended Model ({st.session_state.auto_model_infer})", value=True)
user_model_infer = ""
if not use_auto_infer:
    user_model_infer = st.text_input("Enter a model name for inference")

# 🔧 Limited to FP16 and INT4 only
precision_infer = st.selectbox("Select Quantization for Inference", ["FP16", "INT4"])

if st.button("🚀 Proceed to Inference"):
    st.session_state.model_name = user_model_infer.strip() if not use_auto_infer else st.session_state.auto_model_infer
    st.session_state.precision_choice = precision_infer
    st.success(f"Model `{st.session_state.model_name}` selected with {precision_infer} quantization.")

    with st.spinner("Downloading and converting to GGUF..."):
        out_path = download_and_convert_to_gguf(
            model_repo=st.session_state.model_name,
            quant=st.session_state.precision_choice.lower(),
            hf_token=st.secrets["HF_TOKEN"] if "HF_TOKEN" in st.secrets else None
        )
        if out_path:
            st.session_state.gguf_path = out_path
            st.success(f"GGUF model ready at: {out_path}")
            st.switch_page("pages/4_inference.py")
        else:
            st.error("Failed to download or convert the model to GGUF.")

st.markdown("---")

# ─── Option 2: Fine-Tune First ─────────────────────────
st.subheader("Option 2: Fine-Tune Then Use")
use_auto_ft = st.checkbox(f"Use Auto-Recommended Fine-Tune Model ({st.session_state.auto_model_ft})", value=True)
user_model_ft = ""
if not use_auto_ft:
    user_model_ft = st.text_input("Enter a model name for fine-tuning")

# 🔧 Limited to FP16 and INT4 only
precision_ft = st.selectbox("Select Quantization for Fine-Tuning", ["FP16", "INT4"], key="ft_precision")

if st.button("🧪 Start Fine-Tuning"):
    st.session_state.model_name = user_model_ft.strip() if not use_auto_ft else st.session_state.auto_model_ft
    st.session_state.precision_choice = precision_ft
    st.success(f"Started fine-tuning `{st.session_state.model_name}` with {precision_ft}. (placeholder logic)")
    st.switch_page("pages/4_inference.py")

# ─── Back Navigation ───────────────────────────────────
if st.button("⬅️ Back to Setup"):
    st.switch_page("pages/2_setup_recommendation.py")
