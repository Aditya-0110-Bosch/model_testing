import streamlit as st
import os
from dotenv import load_dotenv
from shared_resources import (
    MODELS,
    extract_text_from_files,
    clean_response,
    create_chat_stream,
    AZURE_API_KEY
)

# =========================================================
# ENV
# =========================================================
load_dotenv()
API_KEY = AZURE_API_KEY

if not API_KEY:
    st.error("AZURE_OPENAI_API_KEY not found in .env")
    st.stop()

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="OrgGPT", page_icon="🤖", layout="wide")

st.markdown("""
<style>
.chat-message { padding: 1rem; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("⚙️ Settings")

model_name = st.sidebar.selectbox("Select Model", MODELS.keys())
cfg = MODELS[model_name]

temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.7)
max_tokens = st.sidebar.slider("Max Output Tokens", 500, 8000, 2000)

system_prompt = st.sidebar.text_area(
    "System Prompt",
    "You are a helpful AI assistant for our organization."
)

uploaded_files = st.sidebar.file_uploader(
    "📎 Upload files (PDF / TXT)", accept_multiple_files=True
)

if st.sidebar.button("🧹 New Conversation"):
    st.session_state.messages = [
        {"role": "system", "content": system_prompt}
    ]
    st.rerun()

# =========================================================
# FILE CONTEXT
# =========================================================
file_context = extract_text_from_files(uploaded_files) if uploaded_files else ""

# =========================================================
# SESSION STATE
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": system_prompt}
    ]

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<h1 style="text-align:center;">🤖 OrgGPT</h1>
<p style="text-align:center; color:gray;">
Unified AI Assistant (Azure OpenAI + Claude)
</p>
""", unsafe_allow_html=True)

# =========================================================
# CHAT HISTORY
# =========================================================
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

prompt = st.chat_input("Ask something...")

# =========================================================
# CHAT EXECUTION
# =========================================================
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("💭 *Generating response…*")

        try:
            full_response = ""
            stream = create_chat_stream(
                model_name,
                st.session_state.messages,
                temperature,
                max_tokens,
                file_context
            )
            
            for chunk in stream:
                full_response += chunk
                placeholder.markdown(
                    clean_response(full_response, cfg.get("strip_think", False))
                )

        except Exception as e:
            full_response = f"❌ Error: {e}"
            placeholder.error(full_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
