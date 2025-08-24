import streamlit as st
from llama_infer import LlamaInference
import math
import os

def compute_confidence(token_logprobs):
    if not token_logprobs or len(token_logprobs) == 0:
        return 0.0
    avg_logprob = sum(token_logprobs) / len(token_logprobs)
    return math.exp(avg_logprob)

# Prompt builder for Llama 3/3.1 chat/instruct models
def build_llama3_prompt(history):
    prompt = "<|begin_of_text|>"
    for role, msg in history:
        if role == "user":
            prompt += "<|start_header_id|>user<|end_header_id|>\n" + msg.strip() + "\n"
        elif role == "assistant":
            prompt += "<|start_header_id|>assistant<|end_header_id|>\n" + msg.strip() + "\n"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
    return prompt

st.title("🧠 HALO Chatbot (Llama.cpp Minimal Output)")

# Model path input
model_path = st.text_input(
    "Path to your GGUF model",
    value=st.session_state.get("model_path", "converted/meta-llama_Llama-3.1-8B.q4_k_m.gguf")
)
st.session_state["model_path"] = model_path

if not os.path.exists(model_path):
    st.warning("⚠️ Model path not found. Please check and try again.")
    st.stop()

# Confidence threshold slider
threshold = st.slider(
    "Confidence threshold for fallback (0=always trust, 1=almost never trust)",
    min_value=0.0, max_value=1.0, value=st.session_state.get("threshold", 0.0), step=0.01
)
st.session_state["threshold"] = threshold

# Model loading (cached)
@st.cache_resource
def load_model(model_path):
    return LlamaInference(model_path)

llama = load_model(model_path)

# Chat state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input (form style to prevent accidental double send)
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", key="user_input", placeholder="Type your message…")
    submitted = st.form_submit_button("Send")

# Handle user submit
if submitted and user_input.strip():
    st.session_state.chat_history.append(("user", user_input.strip()))

    # Build prompt for Llama-3/3.1
    prompt = build_llama3_prompt(st.session_state.chat_history)

    # Run inference with correct stop sequence
    response, token_logprobs = llama.infer(
        prompt, max_tokens=256, stop=["<|start_header_id|>user<|end_header_id|>"]
    )
    confidence = compute_confidence(token_logprobs)

    st.session_state.chat_history.append(("assistant", response.strip()))
    # For debugging, you can add: f"{response.strip()} (conf: {confidence:.2f})"

# Chat display (user right, assistant left)
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)