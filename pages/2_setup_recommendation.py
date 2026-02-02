import os
import subprocess
import streamlit as st
from hardware_recommend import MODEL_CATALOG
from pathlib import Path

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

# ─── Repo Mapping ─────────────────────────────────────
REPO_MAP = {m["name"]: m["repo_id"] for m in MODEL_CATALOG}

# ─── UI ────────────────────────────────────────────────
st.markdown("Upload one or more documents and run hardware detection to get model suggestions.")

uploaded_files = st.file_uploader("Upload Document(s) (PDF, max 10GB total)", type=["txt"], accept_multiple_files=True)
if uploaded_files:
    st.session_state.uploaded_files = uploaded_files
    # Combine all uploaded txt files into one train.txt
    train_dir = Path("train_data")
    train_dir.mkdir(exist_ok=True)
    train_path = train_dir / "train.txt"

    with train_path.open("w", encoding="utf-8") as out_f:
        for uf in uploaded_files:
            # uf is a Streamlit UploadedFile
            content = uf.read()
            try:
                text = content.decode("utf-8")
            except AttributeError:
                # content is already str
                text = content
            out_f.write(text)
            out_f.write("\n")

    # Save path for Step 3 (LoRA fine-tuning)
    st.session_state.train_data_path = str(train_path)
    st.success(f"✅ Training data saved to {train_path}")


hf_token = st.text_input("Hugging Face Access Token", type="password")
st.session_state.hf_token = hf_token.strip()

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
                model_label = line.split(":")[-1].strip().split("@")[0].strip()
                st.session_state.auto_model_infer = REPO_MAP.get(model_label, model_label)
            if "Recommended for QLoRA tuning" in line or "Recommended for LoRA tuning" in line:
                model_label = line.split(":")[-1].strip().split("@")[0].strip()
                st.session_state.auto_model_ft = REPO_MAP.get(model_label, model_label)

if st.session_state.auto_model_infer:
    st.success(f"Auto-Recommended for Inference: {st.session_state.auto_model_infer}")
    st.success(f"Auto-Recommended for Fine-Tuning: {st.session_state.auto_model_ft}")

if uploaded_files and hf_token.strip() and st.session_state.auto_model_infer:
    if st.button("➡️ Proceed to Model Selection"):
        st.switch_page("pages/3_model_selection.py")

if st.button("d⬅️ Back to Start"):
    st.switch_page("main.py")
