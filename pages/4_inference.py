import os
import streamlit as st

st.set_page_config(page_title="Step 4 – Inference", layout="centered")
st.title("🤖 Step 4: Run Inference")

if "model_name" not in st.session_state or "precision_choice" not in st.session_state:
    st.warning("Model not selected. Please go back to Step 3.")
    st.stop()

model_name = st.session_state.model_name
precision = st.session_state.precision_choice

st.markdown(f"**Model Selected:** `{model_name}`")
st.markdown(f"**Precision Mode:** `{precision}`")

# ─── Upload Model Checkpoint (.gguf or LoRA) ─────────────────────────────
st.markdown("### Upload Your GGUF or LoRA Checkpoint")
checkpoint_file = st.file_uploader("Upload model checkpoint file", type=["gguf", "bin", "safetensors"], key="checkpoint")

# ─── Input Prompt ────────────────────────────────────────────────────────
user_input = st.text_area("Enter your input prompt", height=200)

# ─── Dummy Inference (Replace with llama.cpp later) ──────────────────────
if st.button("🔍 Run Inference"):
    if checkpoint_file is None:
        st.warning("Please upload a model checkpoint file first.")
    elif not user_input.strip():
        st.warning("Please enter a prompt.")
    else:
        st.info("Running inference...")

        # Placeholder result
        result = f"[Placeholder Output]\nModel: {model_name}\nPrecision: {precision}\nPrompt: {user_input}"
        st.success("✅ Inference complete!")
        st.code(result, language="text")

# ─── Back Navigation ──────────────────────────────────────────────────────
if st.button("⬅️ Back to Model Selection"):
    st.switch_page("pages/3_model_selection.py")
