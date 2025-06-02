# File: streamlit_app.py
import os
import torch
import streamlit as st
from psych_model import PsychologistAssistant

st.set_page_config(
    page_title="🧠 AI Psychologist Assistant",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .chat-container { max-width: 700px; margin: auto; }
    .user-box { background-color:#d1e7dd; padding:0.75em; border-radius:15px; margin:0.5em 0; font-size:1rem; }
    .assistant-box { background-color:#f8d7da; padding:0.75em; border-radius:15px; margin:0.5em 0; font-size:1rem; }
    .stream-container { background-color:#f7f9fa; padding:1em; border-radius:10px; margin-top:1em; }
    </style>
    <div class='chat-container'>
        <h1 style='text-align: center;'>🧠 AI Psychologist Assistant</h1>
        <p style='text-align: center; color: gray;'>Your personal, low-resource emotional support</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "assistant" not in st.session_state:
    model_path = os.getenv(
        "LORA_MODEL_PATH", r"fine_tuned_weights\Llama-2-7b-chat-hf-finetune"
    )
    st.session_state.assistant = PsychologistAssistant(model_path)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.form(key="input_form", clear_on_submit=True):
    prompt = st.text_input("How are you feeling today?", key="prompt_input")
    submit_button = st.form_submit_button("Send")

if submit_button and prompt:
    st.session_state.chat_history.append(("user", prompt))
    with st.container():
        st.markdown(
            f"<div class='user-box'>{prompt}</div>", unsafe_allow_html=True)

    # Stream assistant response
    with st.container():
        st.markdown(
            "<div class='assistant-box'><strong>AI:</strong></div>", unsafe_allow_html=True)
        streamer = st.session_state.assistant.respond_stream(
            prompt, temperature=0.6, max_new_tokens=80)
        response_placeholder = st.empty()
        full_response = ""
        for new_token in streamer:
            full_response += new_token
            response_placeholder.markdown(
                f"<div class='stream-container'>{full_response}</div>",
                unsafe_allow_html=True,
            )
        st.session_state.chat_history.append(("assistant", full_response))

if st.session_state.chat_history:
    for role, message in st.session_state.chat_history:
        if role == "user":
            st.markdown(
                f"<div class='user-box'>{message}</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div class='assistant-box'>{message}</div>", unsafe_allow_html=True)

st.markdown(
    """
    ---
    <p style='text-align: center; color: lightgray;'>Built with ❤️ using Streamlit and Transformers</p>
    """,
    unsafe_allow_html=True,
)
