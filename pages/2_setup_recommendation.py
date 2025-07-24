import os
import subprocess
import streamlit as st

st.set_page_config(page_title="Step 2 – Setup & Hardware Recommendation", layout="centered")
st.title("🛠️ Step 2: Upload Documents & Hardware Recommendation")

# ─── Initialize Session State ─────────────────────────
if "stage" not in st.session_state:
    st.session_state.stage = "setup"
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "auto_model_infer" not in st.session_state:
    st.session_state.auto_model_infer = ""
if "auto_model_ft" not in st.session_state:
    st.session_state.auto_model_ft = ""

# ─── UI ────────────────────────────────────────────────
st.markdown("Upload one or more documents and run hardware detection to get model suggestions.")

uploaded_files = st.file_uploader("Upload Document(s) (PDF, max 10GB total)", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    st.session_state.uploaded_files = uploaded_files

hf_token = st.text_input("Hugging Face Access Token", type="password")
user_model = st.text_input("Optional: Hugging Face Model Name (e.g. Qwen/Qwen3-32B)")

if st.button("🔍 Run Hardware Recommendation"):
    if not uploaded_files:
        st.warning("Please upload at least one document.")
    elif not hf_token.strip():
        st.warning("Please enter your Hugging Face token.")
    else:
        env = os.environ.copy()
        env["HF_TOKEN"] = hf_token.strip()

        command = ["python", "hardware_recommend.py"]
        if user_model.strip():
            command.append(user_model.strip())

        result = subprocess.run(command, capture_output=True, text=True, env=env)
        output = result.stdout or result.stderr
        st.code(output, language="bash")

        for line in output.splitlines():
            if "Recommended for inference" in line:
                st.session_state.auto_model_infer = line.split(":")[-1].strip()
            if "Recommended for LoRA tuning" in line:
                st.session_state.auto_model_ft = line.split(":")[-1].strip()

if st.session_state.auto_model_infer:
    st.success(f"Auto-Recommended for Inference: {st.session_state.auto_model_infer}")
    st.success(f"Auto-Recommended for Fine-Tuning: {st.session_state.auto_model_ft}")

if uploaded_files and hf_token.strip() and st.session_state.auto_model_infer:
    if st.button("➡️ Proceed to Model Selection"):
        st.switch_page("pages/3_model_selection.py")

if st.button("⬅️ Back to Start"):
    st.switch_page("main.py")
