import os
import subprocess
import streamlit as st
import random

st.set_page_config(page_title="HALO Architecture Demo UI", layout="centered")
st.title("HALO Architecture Demo UI")

# ─── Page State ─────────────────────────────────────────────
if "stage" not in st.session_state:
    st.session_state.stage = "setup"
if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = []
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "model_name" not in st.session_state:
    st.session_state.model_name = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "auto_model_infer" not in st.session_state:
    st.session_state.auto_model_infer = "OPT-13B"
if "auto_model_ft" not in st.session_state:
    st.session_state.auto_model_ft = "OPT-13B"
if "precision_choice" not in st.session_state:
    st.session_state.precision_choice = "INT8"
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.7
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ─── Helper: Back Navigation ─────────────────────────────────
def back_button(prev_stage):
    if st.button("⬅️ Back"):
        st.session_state.stage = prev_stage

# ─── Stage 1: Setup ──────────────────────────────────────────
if st.session_state.stage == "setup":
    st.header("1. Upload Documents and Configure Model")

    uploaded_files = st.file_uploader("Upload Documents (PDF, up to 10GB total)", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files

    hf_token = st.text_input("Hugging Face Access Token", type="password")
    user_model = st.text_input("Optional Hugging Face Model (e.g. Qwen/Qwen3-32B)", value="")

    if st.button("Run Hardware Recommendation"):
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
            reco_output = result.stdout or result.stderr
            st.code(reco_output, language="bash")

            for line in reco_output.splitlines():
                if "Recommended for inference" in line:
                    st.session_state.auto_model_infer = line.split(":")[-1].strip()
                if "Recommended for QLoRA tuning" in line:
                    st.session_state.auto_model_ft = line.split(":")[-1].strip()

    if uploaded_files and hf_token.strip():
        if st.button("Next: Proceed to Model Selection"):
            st.session_state.stage = "model_selection"

# ─── Stage 2: Model Selection ───────────────────────────────
elif st.session_state.stage == "model_selection":
    st.header("2. Choose What to Do Next")

    st.subheader("Option 1: Direct Inference (no fine-tuning)")
    use_auto_infer = st.checkbox(f"Use Auto-Recommended Model for Inference ({st.session_state.auto_model_infer})", value=True)
    user_model_infer = ""
    if not use_auto_infer:
        user_model_infer = st.text_input("Enter Model Name for Inference")

    precision_infer = st.selectbox("Select Quantized Precision for Inference", ["FP16", "INT8", "INT4"])

    if st.button("Proceed with Inference"):
        st.session_state.model_name = user_model_infer.strip() if not use_auto_infer else st.session_state.auto_model_infer
        st.session_state.precision_choice = precision_infer
        st.success(f"Loading model `{st.session_state.model_name}` @ {precision_infer} for inference...")
        st.session_state.stage = "chat_ui"

    st.markdown("---")

    st.subheader("Option 2: Fine-Tune Then Inference")
    use_auto_ft = st.checkbox(f"Use Auto-Recommended Model for Fine-Tuning ({st.session_state.auto_model_ft})", value=True)
    user_model_ft = ""
    if not use_auto_ft:
        user_model_ft = st.text_input("Enter Model Name for Fine-Tuning")

    precision_ft = st.selectbox("Select Quantized Precision for Fine-Tuning", ["FP16", "INT8", "INT4"], key="ft_precision")

    if st.button("Start Fine-Tuning"):
        st.session_state.model_name = user_model_ft.strip() if not use_auto_ft else st.session_state.auto_model_ft
        st.session_state.precision_choice = precision_ft
        st.success(f"Fine-tuning started on model `{st.session_state.model_name}` with {precision_ft}... (placeholder logic)")
        st.session_state.stage = "chat_ui"

    back_button("setup")

# ─── Stage 3: Chatbot Inference UI ───────────────────────────
elif st.session_state.stage == "chat_ui":
    st.header("3. AI Agent Chat Interface")
    st.write(f"Using model: `{st.session_state.model_name}` @ {st.session_state.precision_choice}")

    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    user_input = st.chat_input("Ask me anything")
    if user_input:
        st.chat_message("user").write(user_input)
        confidence = random.uniform(0.4, 0.95)
        if confidence < st.session_state.threshold:
            response = f"Big model fallback response to: '{user_input}'"
            used_big = True
        else:
            response = f"Local model response to: '{user_input}'"
            used_big = False

        st.chat_message("assistant").write(response)
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        st.session_state.feedback_log.append({
            "query": user_input,
            "answer": response,
            "confidence": confidence,
            "used_big_model": used_big,
            "rating": "pending"
        })

    with st.expander("⚙️ Threshold Settings"):
        st.session_state.threshold = st.slider("Confidence threshold (below this, fallback to big model)", 0.0, 1.0, st.session_state.threshold)

    back_button("model_selection")
