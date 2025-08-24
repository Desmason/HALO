import streamlit as st
import os
from llama_infer import LlamaInference

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
    # --- In the future, call the OpenAI API here! ---
    # Example (for real use, see OpenAI docs):
    #   import openai
    #   openai.api_key = "YOUR-KEY"
    #   response = openai.ChatCompletion.create(
    #       model="gpt-3.5-turbo",
    #       messages=messages,
    #   )
    #   return response['choices'][0]['message']['content']
    # For now, just this placeholder:
    return "This is where the OpenAI response would go."


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

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Accept user input ----
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    llama_prompt = build_llama3_prompt(st.session_state.messages)
    llama_response, token_logprobs = llama.infer(
        llama_prompt, max_tokens=256, stop=["<|start_header_id|>user<|end_header_id|>"]
    )
    confidence = compute_confidence(token_logprobs)  # Always 0.70

    if confidence >= threshold:
        response = f"(Local Llama, conf={confidence:.2f}) {llama_response.strip()}"
    else:
        openai_response = get_openai_response(st.session_state.messages)
        response = f"(OpenAI fallback! conf={confidence:.2f}) {openai_response}"

    st.session_state.messages.append({"role": "assistant", "content": response})

# ---- Render chat with thumbs up/down ----
if "ratings" not in st.session_state:
    st.session_state.ratings = {}  # Key: idx, Value: "up" or "down"

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            col1, col2, col3 = st.columns([1, 1, 8])
            with col1:
                if st.button("👍", key=f"thumbs_up_{idx}"):
                    st.session_state.ratings[idx] = "up"
            with col2:
                if st.button("👎", key=f"thumbs_down_{idx}"):
                    st.session_state.ratings[idx] = "down"
            with col3:
                if idx in st.session_state.ratings:
                    if st.session_state.ratings[idx] == "up":
                        st.markdown("**<span style='color:green'>👍 Thank you!</span>**", unsafe_allow_html=True)
                    elif st.session_state.ratings[idx] == "down":
                        st.markdown("**<span style='color:#d9534f'>👎 Feedback noted.</span>**", unsafe_allow_html=True)
