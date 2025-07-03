import os

import requests
import streamlit as st

# Set API URL from environment variable or fallback
API_URL = os.getenv("DOC_CHAT_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Doc Chat", layout="centered")
st.title("📄 Doc Chat: PDF Q&A Demo")
st.markdown(
    """
Welcome! Upload a PDF, ask questions, and get answers with context from the document. <br>
**Powered by doc-chat-api**
""",
    unsafe_allow_html=True,
)

# Session state for file_id
if "file_id" not in st.session_state:
    st.session_state["file_id"] = None
if "file_name" not in st.session_state:
    st.session_state["file_name"] = None

# --- PDF Upload ---
st.header("1. Upload your PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
    with st.spinner("Uploading..."):
        try:
            res = requests.post(f"{API_URL}/api/upload", files=files)
            res.raise_for_status()
            file_id = res.json().get("file_id")
            if file_id:
                st.session_state["file_id"] = file_id
                st.session_state["file_name"] = uploaded_file.name
                st.success(f"Uploaded '{uploaded_file.name}'! File ID: {file_id}")
            else:
                st.error("Upload failed: No file_id returned.")
        except Exception as e:
            st.error(f"Upload failed: {e}")

# --- Q&A Section ---
st.header("2. Ask a question")
if st.session_state["file_id"]:
    st.markdown(
        f"**Current file:** `{st.session_state['file_name']}` (ID: `{st.session_state['file_id']}`)"
    )
    question = st.text_input("Enter your question about the PDF:")
    if st.button("Ask") and question:
        with st.spinner("Getting answer..."):
            try:
                payload = {"file_id": st.session_state["file_id"], "question": question}
                res = requests.post(f"{API_URL}/api/chat", json=payload)
                res.raise_for_status()
                data = res.json()
                answer = data.get("answer", "No answer returned.")
                context_chunks = data.get("context_chunks", [])
                st.markdown("### 💬 Answer:")
                st.markdown(f"> {answer}")
                if context_chunks:
                    st.markdown("---")
                    st.markdown("#### 🔍 Context Chunks:")
                    for i, chunk in enumerate(context_chunks, 1):
                        st.markdown(
                            f"""**Chunk {i}:**\n```
{chunk}
```"""
                        )
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("Please upload a PDF to begin.")

st.markdown("---")
st.caption(
    "Made with ❤️ for recruiters and devs. [GitHub](https://github.com/TimB808/doc-chat-api)"
)
