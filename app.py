"""
Career Coach Agent - Streamlit UI

A single-agent AI application that helps Nigerian tech students and
early-career professionals with career guidance, CV feedback, and
job search strategy.
"""

import streamlit as st
from agent import CareerCoachAgent
from tools.cv_analyzer import extract_cv_text, summarize_cv_stats

st.set_page_config(
    page_title="Career Coach Agent",
    page_icon="🎯",
    layout="centered",
)

# ---------- Session State Init ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "cv_text" not in st.session_state:
    st.session_state.cv_text = ""

if "agent" not in st.session_state:
    api_key = st.secrets.get("GROQ_API_KEY", "")
    if api_key:
        st.session_state.agent = CareerCoachAgent(api_key=api_key)
    else:
        st.session_state.agent = None

# ---------- Sidebar ----------
with st.sidebar:
    st.title("🎯 Career Coach Agent")
    st.caption("An AI career coach for Nigerian tech students & early-career professionals.")

    st.divider()
    st.subheader("📄 Upload Your CV")
    uploaded_file = st.file_uploader("PDF only", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Analyze CV", use_container_width=True):
            with st.spinner("Reading your CV..."):
                pdf_bytes = uploaded_file.read()
                cv_text = extract_cv_text(pdf_bytes)

                if cv_text.startswith("ERROR"):
                    st.error(cv_text)
                else:
                    st.session_state.cv_text = cv_text
                    stats = summarize_cv_stats(cv_text)
                    st.success(f"CV loaded ({stats['word_count']} words). The agent will now use it for tailored advice.")

    if st.session_state.cv_text:
        st.info("✅ CV is loaded and being used for context.")
        if st.button("Clear CV", use_container_width=True):
            st.session_state.cv_text = ""
            st.rerun()

    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "Built with Groq API · RAG-grounded career knowledge base · "
        "Live web search · CV parsing"
    )

# ---------- Main Chat Interface ----------
st.title("Career Coach Agent")
st.caption("Ask me about career paths, CV feedback, skill gaps, or the job market. Upload your CV in the sidebar for tailored advice.")

if st.session_state.agent is None:
    st.error(
        "⚠️ No API key found. Add your GROQ_API_KEY in Streamlit secrets "
        "(Settings → Secrets) to use this app."
    )
    st.stop()

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask about your career, CV, or job search...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_text = st.session_state.agent.respond(
                conversation_history=st.session_state.messages,
                cv_context=st.session_state.cv_text,
            )
            st.markdown(response_text)

    st.session_state.messages.append({"role": "assistant", "content": response_text})
