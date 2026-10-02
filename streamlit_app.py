import streamlit as st
import requests
import time

# ===============================
# Backend URL
# ===============================
BACKEND_URL = "http://localhost:8000"


# ===============================
# Page Configuration
# ===============================
st.set_page_config(
    page_title="NCERT AI Tutor",
    page_icon="📚",
    layout="wide"
)


# ===============================
# Custom CSS for Better UI
# ===============================
st.markdown(
    """
<style>
.chat-container {
    border-radius: 10px;
    padding: 10px;
}

.source-box {
    background-color: #f5f5f5;
    padding: 10px;
    border-radius: 8px;
}
</style>
""",
    unsafe_allow_html=True
)


# ===============================
# Session State for Chat History
# ===============================
if "messages" not in st.session_state:
    st.session_state.messages = []


# ===============================
# Sidebar
# ===============================
with st.sidebar:

    st.title("📚 NCERT AI Tutor")

    st.markdown(
        """
This AI tutor answers questions from **NCERT textbooks (Class 6–12)** using Retrieval Augmented Generation (RAG).

### Features
✔ AI Generated Answers  
✔ NCERT Document Retrieval  
✔ Page References  
✔ Chat History  

### Usage
Type a question and press **Enter**.
"""
    )

    st.divider()

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# ===============================
# App Title
# ===============================
st.title("📚 NCERT AI Tutor")
st.caption("Ask questions from NCERT textbooks and get AI-generated answers with references.")


# ===============================
# Display Previous Messages
# ===============================
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])

        if message.get("docs"):

            st.markdown("📄 **Sources**")

            for doc in message["docs"]:
                with st.expander(f"Page {doc['page']}"):
                    st.write(doc["snippet"])
                    st.markdown(f"[Open Document]({doc['link']})")


# ===============================
# Chat Input
# ===============================
question = st.chat_input("Ask a question from NCERT...")


# ===============================
# When User Asks Question
# ===============================
if question:

    # Show user message
    st.chat_message("user").write(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # Assistant response container
    with st.chat_message("assistant"):

        message_placeholder = st.empty()

        with st.spinner("Searching NCERT knowledge base..."):

            try:
                response = requests.post(
                    f"{BACKEND_URL}/ask",
                    json={"question": question}
                )

            except Exception as e:
                st.error("⚠ Unable to connect to backend server")
                st.stop()


        if response.status_code == 200:

            data = response.json()

            answer = data.get("answer", "No answer returned.")
            docs = data.get("retrieved_documents", [])


            # Streaming answer effect
            full_response = ""

            for word in answer.split():
                full_response += word + " "
                time.sleep(0.02)
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)


            # Show sources
            if docs:

                st.markdown("### 📄 Sources")

                for doc in docs:

                    with st.expander(f"Page {doc['page']}"):

                        st.markdown("**Snippet:**")
                        st.write(doc["snippet"])

                        st.markdown(
                            f"[Open Document]({doc['link']})"
                        )


            # Save assistant message
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "docs": docs
                }
            )


        else:
            st.error(f"Backend error: {response.status_code}")