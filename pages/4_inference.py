import streamlit as st
import os
import json
import time
from datetime import datetime
from llama_infer import LlamaInference
from confidence_math import (
    compute_hybrid_confidence,
    decide_route,
    compute_confidence,
    compute_confidence_entropy,
)

# =========================
# ---- LOGGING HELPERS ----
# =========================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "halo_chat.json")

def load_logs():
    return json.load(open(LOG_FILE, "r", encoding="utf-8")) if os.path.exists(LOG_FILE) else []

def save_logs(logs):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

def log_exchange(user_query: str, response: str):
    logs = load_logs()
    record = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "user_query": user_query,
        "response": response,
        "rating": 0,
    }
    logs.append(record)
    save_logs(logs)
    return len(logs) - 1

def update_rating(record_idx: int, rating: int):
    logs = load_logs()
    if 0 <= record_idx < len(logs):
        logs[record_idx]["rating"] = rating
        save_logs(logs)

# =========================
# ---- PROMPT HELPERS -----
# =========================
def build_llama3_prompt(messages):
    prompt = "<|begin_of_text|>"
    for m in messages:
        role = m["role"]
        content = m["content"].strip()
        if role == "user":
            prompt += f"<|start_header_id|>user<|end_header_id|>\n{content}\n"
        elif role == "assistant":
            prompt += f"<|start_header_id|>assistant<|end_header_id|>\n{content}\n"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
    return prompt

def get_openai_response(messages):
    # Placeholder
    return "This is where the OpenAI response would go."

# =========================
# --------- UI ------------
# =========================
st.title("🧠 HALO Chatbot (Llama.cpp + OpenAI fallback demo)")

# ---- Reset session-state button ----
if st.button("🔄 Reset model paths"):
    for key in ["base_model_path", "lora_path"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# ---- Inference mode ----
mode = st.radio(
    "Inference mode",
    ["Direct base model", "Fine-tuned (base + LoRA)"],
    horizontal=True,
)

# Default clean values (only if not set)
if "base_model_path" not in st.session_state:
    st.session_state["base_model_path"] = ""
if "lora_path" not in st.session_state:
    st.session_state["lora_path"] = ""

# Input fields
if mode == "Direct base model":
    base_model_path = st.text_input(
        "Path to your base GGUF model",
        value=st.session_state["base_model_path"],
        key="base_model_path"
    )
    lora_path = None
else:
    base_model_path = st.text_input(
        "Base GGUF model path",
        value=st.session_state["base_model_path"],
        key="base_model_path"
    )
    lora_path = st.text_input(
        "LoRA GGUF adapter path",
        value=st.session_state["lora_path"],
        key="lora_path"
    )

# ---- Validate base path ----
if not os.path.exists(base_model_path) or base_model_path.strip() == "":
    st.warning(f"⚠️ Base model not found:\n`{base_model_path}`")
    st.stop()

# ---- Validate LoRA path ----
if mode == "Fine-tuned (base + LoRA)":
    if lora_path is None or lora_path.strip() == "" or not os.path.exists(lora_path):
        st.warning(f"⚠️ LoRA adapter not found:\n`{lora_path}`")
        st.stop()

# ---- Privacy slider φ ----
phi_raw = st.slider("Privacy (0 = all OpenAI, 100 = all local)", 0, 100, 50, 5)
phi = phi_raw / 100.0
st.session_state["privacy"] = phi

# ---- Load model (cached) ----
@st.cache_resource
def load_model(base, lora):
    return LlamaInference(model_path=base, lora_path=lora)

llama = load_model(base_model_path, lora_path)

# ---- Initialize chat session ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ratings" not in st.session_state:
    st.session_state.ratings = {}
if "record_ids" not in st.session_state:
    st.session_state.record_ids = {}

# ---- Accept user input ----
if prompt := st.chat_input("Type your message..."):
    t0 = time.time()
    st.session_state.messages.append({"role": "user", "content": prompt})

    llama_prompt = build_llama3_prompt(st.session_state.messages)
    llama_response, token_logprobs, top_logprobs = llama.infer(
        llama_prompt,
        max_tokens=256,
        stop=["<|start_header_id|>user<|end_header_id|>"],
        logprobs_k=5,
    )

    # ---- HALO Confidence Logic ----
    conf_ppl, _ = compute_confidence(token_logprobs)
    conf_ent, _ = compute_confidence_entropy(top_logprobs)
    hybrid = compute_hybrid_confidence(token_logprobs, top_logprobs)

    route = decide_route(hybrid, phi, deterministic=True)
    took_ms = int((time.time() - t0) * 1000)

    metrics = (
        f"📊 conf_ppl={conf_ppl:.3f} | conf_ent={conf_ent:.3f} | "
        f"hybrid={hybrid:.3f} | φ={phi:.2f} | route={route} | ⏱️ {took_ms} ms"
    )

    # ---- Choose response ----
    if route == "local":
        final_text = llama_response.strip()
    else:
        final_text = get_openai_response(st.session_state.messages)

    response = f"{final_text}\n\n{metrics}"

    st.session_state.messages.append({"role": "assistant", "content": response})
    record_idx = log_exchange(prompt, final_text)
    st.session_state.record_ids[len(st.session_state.messages) - 1] = record_idx

# ---- Render messages ----
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            col1, col2, col3 = st.columns([1, 1, 8])
            with col1:
                if st.button("👍", key=f"up_{idx}"):
                    st.session_state.ratings[idx] = 1
                    update_rating(st.session_state.record_ids[idx], 1)
            with col2:
                if st.button("👎", key=f"down_{idx}"):
                    st.session_state.ratings[idx] = -1
                    update_rating(st.session_state.record_ids[idx], -1)
            with col3:
                if idx in st.session_state.ratings:
                    if st.session_state.ratings[idx] == 1:
                        st.markdown("**<span style='color:green'>👍 Thank you!</span>**", unsafe_allow_html=True)
                    elif st.session_state.ratings[idx] == -1:
                        st.markdown("**<span style='color:#d9534f'>👎 Feedback noted.</span>**", unsafe_allow_html=True)
