import streamlit as st
import os
import json
import time
from datetime import datetime
from llama_infer import LlamaInference

# =========================
# ---- LOGGING HELPERS ----
# =========================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "halo_chat.json")

def load_logs():
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_logs(logs):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

def log_exchange(user_query: str, response: str):
    logs = load_logs()
    record = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "user_query": user_query,
        "response": response,
        "rating": 0   # default not rated
    }
    logs.append(record)
    save_logs(logs)
    return len(logs) - 1  # index of this record

def update_rating(record_idx: int, rating: int):
    logs = load_logs()
    if 0 <= record_idx < len(logs):
        logs[record_idx]["rating"] = rating
        save_logs(logs)

# =========================
# ---- CONFIDENCE ----------
# =========================
def compute_confidence(token_logprobs):
    # Always return 0.70 for demo/fallback logic
    return 0.70

def build_llama3_prompt(messages):
    prompt = "<|begin_of_text|>"
    for m in messages:
        if m["role"] == "user":
            prompt += "<|start_header_id|>user<|end_header_id|>\n" + m["content"].strip() + "\n"
        elif m["role"] == "assistant":
            prompt += "<|start_header_id|>assistant<|end_header_id|>\n" + m["content"].strip() + "\n"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
    return prompt

def get_openai_response(messages):
    # Placeholder for a real OpenAI call
    return "This is where the OpenAI response would go."

# =========================
# --------- UI ------------
# =========================
st.title("🧠 HALO Chatbot (Llama.cpp + OpenAI fallback demo)")

model_path = st.text_input(
    "Path to your GGUF model",
    value=st.session_state.get("model_path", "converted/meta-llama_Llama-3.1-8B-Instruct.q4_k_m.gguf")
)
st.session_state["model_path"] = model_path

if not os.path.exists(model_path):
    st.warning("⚠️ Model path not found. Please check and try again.")
    st.stop()

threshold = st.slider(
    "Confidence threshold for OpenAI fallback (0=always trust, 1=never trust Llama)",
    min_value=0.0, max_value=1.0, value=0.70, step=0.01
)
st.session_state["threshold"] = threshold

@st.cache_resource
def load_model(model_path):
    return LlamaInference(model_path)

llama = load_model(model_path)

# Session state init
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ratings" not in st.session_state:
    st.session_state.ratings = {}   # assistant_idx -> rating
if "record_ids" not in st.session_state:
    st.session_state.record_ids = {}  # assistant_idx -> log index

# ---- Accept user input ----
if prompt := st.chat_input("Type your message..."):
    t0 = time.time()

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Run inference
    llama_prompt = build_llama3_prompt(st.session_state.messages)
    llama_response, token_logprobs = llama.infer(
        llama_prompt, max_tokens=256, stop=["<|start_header_id|>user<|end_header_id|>"]
    )
    confidence = compute_confidence(token_logprobs)

    # Fallback if needed
    if confidence >= threshold:
        provenance = "local"
        final_text = llama_response.strip()
        response = f"(Local Llama, conf={confidence:.2f}) {final_text}"
    else:
        provenance = "openai_fallback"
        final_text = get_openai_response(st.session_state.messages)
        response = f"(OpenAI fallback! conf={confidence:.2f}) {final_text}"

    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Log this exchange (without rating yet)
    record_idx = log_exchange(prompt, final_text)
    st.session_state.record_ids[len(st.session_state.messages) - 1] = record_idx

    took_ms = int((time.time() - t0) * 1000)
    st.caption(f"⏱️ Took {took_ms} ms — provenance: {provenance}")

# ---- Render chat with thumbs up/down ----
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            col1, col2, col3 = st.columns([1, 1, 8])
            with col1:
                if st.button("👍", key=f"thumbs_up_{idx}"):
                    st.session_state.ratings[idx] = 1
                    update_rating(st.session_state.record_ids[idx], 1)
            with col2:
                if st.button("👎", key=f"thumbs_down_{idx}"):
                    st.session_state.ratings[idx] = -1
                    update_rating(st.session_state.record_ids[idx], -1)
            with col3:
                if idx in st.session_state.ratings:
                    if st.session_state.ratings[idx] == 1:
                        st.markdown("**<span style='color:green'>👍 Thank you!</span>**", unsafe_allow_html=True)
                    elif st.session_state.ratings[idx] == -1:
                        st.markdown("**<span style='color:#d9534f'>👎 Feedback noted.</span>**", unsafe_allow_html=True)
