"""
Streamlit web UI for the team RAG system.

Run with:
    streamlit run app.py

Then open http://localhost:8501 in your browser.
"""

import streamlit as st

st.set_page_config(
    page_title="Team Document Assistant",
    page_icon="📚",
    layout="centered",
)

st.title("📚 Team Document Assistant")
st.caption("Ask questions about your team's internal documents.")


@st.cache_resource(show_spinner="Loading AI model (this takes ~60 seconds the first time)...")
def load_chain():
    """Load the RAG chain once and cache it for the session."""
    from rag_chain import RAGChain
    return RAGChain()


def display_sources(sources: list[str]):
    if sources:
        with st.expander("Sources used"):
            for src in sources:
                st.write(f"- {src}")


# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            display_sources(message["sources"])

# Load model (cached after first load)
try:
    chain = load_chain()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer... (may take 30-90 seconds on CPU)"):
            try:
                answer, sources = chain.ask_with_sources(prompt)
            except Exception as e:
                answer = f"Error generating answer: {e}"
                sources = []

        st.markdown(answer)
        display_sources(sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
