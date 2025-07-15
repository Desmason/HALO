import streamlit as st
import random
import os

st.set_page_config(page_title="Step 4 – Inference", layout="centered")
st.title("🤖 Step 4: AI Chatbot Inference")

# ─── Fallback Defaults if Coming From Checkpoint Upload ───────
if "model_name" not in st.session_state:
    st.session_state.model_name = "user-uploaded-checkpoint"
if "precision_choice" not in st.session_state:
    st.session_state.precision_choice = "INT4"
if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = []


# ─── Model Checkpoint Upload ─────────────────────────
st.subheader("📦 Upload Your Fine-Tuned Model Checkpoint")
model_file = st.file_uploader("Upload model checkpoint (e.g., .bin, .pt, .safetensors)", type=["bin", "pt", "safetensors"])

if model_file:
    model_path = os.path.join("temp_models", model_file.name)
    os.makedirs("temp_models", exist_ok=True)
    with open(model_path, "wb") as f:
        f.write(model_file.read())
    st.success(f"✅ Checkpoint uploaded to `{model_path}` (simulation only)")
    st.session_state.model_checkpoint_path = model_path

# ─── Show Model Details ─────────────────────────────
st.markdown(f"**Model in use:** `{st.session_state.model_name}`")
st.markdown(f"**Quantization:** `{st.session_state.precision_choice}`")

# ─── Chat History ───────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])

# ─── Chat Input Box ─────────────────────────────────
user_input = st.chat_input("Ask your model something...")
if user_input:
    st.chat_message("user").write(user_input)

    confidence = random.uniform(0.4, 0.95)
    threshold = st.session_state.get("threshold", 0.7)

    if confidence < threshold:
        reply = f"🔁 Fallback (Big Model): Response to '{user_input}'"
        used_big = True
    else:
        reply = f"✅ Local Model: Response to '{user_input}'"
        used_big = False

    st.chat_message("assistant").write(reply)

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": reply})

    st.session_state.feedback_log.append({
        "query": user_input,
        "answer": reply,
        "confidence": confidence,
        "used_big_model": used_big,
        "rating": "pending"
    })

# ─── Threshold Control ──────────────────────────────
with st.expander("⚙️ Inference Settings"):
    st.session_state.threshold = st.slider("Confidence Threshold", 0.0, 1.0, st.session_state.get("threshold", 0.7))

# ─── Back Navigation ────────────────────────────────
if st.button("⬅️ Back to Model Selection"):
    st.switch_page("pages/3_model_selection.py")
