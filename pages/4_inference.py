import streamlit as st
import os
from llama_infer import LlamaInference  # Your own module
import math

# Helper to build Llama-3/3.1 prompt from message history
def build_llama3_prompt(messages):
    prompt = "<|begin_of_text|>"
    for m in messages:
        if m["role"] == "user":
            prompt += "<|start_header_id|>user<|end_header_id|>\n" + m["content"].strip() + "\n"
        elif m["role"] == "assistant":
            prompt += "<|start_header_id|>assistant<|end_header_id|>\n" + m["content"].strip() + "\n"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
    return prompt

def compute_confidence(token_logprobs):
    if not token_logprobs or len(token_logprobs) == 0:
        return 0.0
    avg_logprob = sum(token_logprobs) / len(token_logprobs)
    return math.exp(avg_logprob)

st.title("🧠 HALO Chatbot")

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

@st.cache_resource
def load_model(model_path):
    return LlamaInference(model_path)

llama = load_model(model_path)

# Initialize chat history in the correct format
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the chat messages from history at the top
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Accept user input at the bottom
if prompt := st.chat_input("Type your message..."):
    # Add user's message to history right away
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user's message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build prompt and get Llama response
    llama_prompt = build_llama3_prompt(st.session_state.messages)
    response, token_logprobs = llama.infer(
        llama_prompt, max_tokens=256, stop=["<|start_header_id|>user<|end_header_id|>"]
    )
    confidence = compute_confidence(token_logprobs)

    # Display assistant response immediately
    with st.chat_message("assistant"):
        st.markdown(response.strip())

    # Add assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": response.strip()})
