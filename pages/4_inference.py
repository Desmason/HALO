import streamlit as st
import os
import json
import time
from datetime import datetime
from llama_infer import LlamaInference
from confidence_math import compute_hybrid_confidence, decide_route,compute_confidence, compute_confidence_entropy



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
        "rating": 0
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

# Model path input
model_path = st.text_input(
    "Path to your GGUF model",
    value=st.session_state.get("model_path", "converted/meta-llama_Llama-3.1-8B-Instruct.q4_k_m.gguf")
)
st.session_state["model_path"] = model_path

if not os.path.exists(model_path):
    st.warning("⚠️ Model path not found. Please check and try again.")
    st.stop()

# Privacy slider φ
phi_raw = st.slider("Privacy (0 = all OpenAI, 100 = all local)", 0, 100, 50, 5)
phi = phi_raw / 100.0
st.session_state["privacy"] = phi

@st.cache_resource
def load_model(model_path):
    return LlamaInference(model_path)

llama = load_model(model_path)

# Init session state
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
    out = llama.infer(
        llama_prompt,
        max_tokens=256,
        stop=["<|start_header_id|>user<|end_header_id|>"],
        logprobs_k=5
    )
    llama_response, token_logprobs, top_logprobs = out

    # Compute hybrid confidence
    conf_ppl, _ = compute_confidence(token_logprobs)
    conf_ent, _ = compute_confidence_entropy(top_logprobs)

    c = compute_hybrid_confidence(token_logprobs, top_logprobs)

    # Decide route
    route = decide_route(c, phi, deterministic=True)

    took_ms = int((time.time() - t0) * 1000)


    metrics = (
        f"📊 conf_ppl={conf_ppl:.3f} | conf_ent={conf_ent:.3f} | "
        f"hybrid={c:.3f} | φ={phi:.2f} | route={route} | ⏱️ {took_ms} ms"
    )


    if route == "local":
        provenance = "local"
        final_text = llama_response.strip()
        response = f"{final_text}\n\n{metrics}"
    else:
        provenance = "openai_fallback"
        final_text = get_openai_response(st.session_state.messages)
        response = f"{final_text}\n\n{metrics}"


    st.session_state.messages.append({"role": "assistant", "content": response})
    record_idx = log_exchange(prompt, final_text)
    st.session_state.record_ids[len(st.session_state.messages) - 1] = record_idx

    
    took_ms = int((time.time() - t0) * 1000)
    '''
    st.caption(
    f"🔎 conf_ppl={conf_ppl:.3f} | conf_ent={conf_ent:.3f} | "
    f"hybrid={c:.3f} | φ={phi:.2f} | route={route} | ⏱️ {took_ms} ms"
)
'''


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
