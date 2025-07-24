# main.py - Launcher Page
import streamlit as st

st.set_page_config(page_title="HALO LLM Toolkit", layout="centered")

st.title("🧠 HALO Framework UI")
st.markdown("""
Welcome to the **HALO (Hierarchical and Adaptive Language Operations)** Framework.

Choose how you'd like to begin:

1. Upload your own model checkpoint and go directly to **Inference**.
2. Go through **hardware-based recommendation** and model setup (then fine-tune or use).
""")

# Optional: Let user specify checkpoint quantization
quant_options = ["FP16", "INT4"]
default_index = 1 if len(quant_options) > 1 else 0
precision = st.selectbox("Select Precision for Your Uploaded Model", quant_options, index=default_index)


col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 Use My Checkpoint (Go to Inference)"):
        st.session_state.model_name = "user-uploaded-checkpoint"
        st.session_state.precision_choice = precision
        st.switch_page("pages/4_inference.py")

with col2:
    if st.button("🛠️ Start from Model Setup"):
        st.switch_page("pages/2_setup_recommendation.py")
