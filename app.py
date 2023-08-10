from datetime import datetime
import json
import streamlit as st
from ingest import LOADER_MAPPING, main as ingest_docs
from routes.overloadPDF import inference
from utils.utils import FakeArgs, SimpleStreamlitCallbackHandler

# Sidebar
st.sidebar.title("⚡ Overload PDF Chat 🤖")


st.sidebar.warning(
    "OverloadChat may produce inaccurate information about people, documents, date or amounts. Please always refer to the original document",
    icon="⚠",
)

with st.expander(label="File uploader"):
    if uploaded_files := st.sidebar.file_uploader(
        "Upload a file",
        # type=[x.replace(".", "") for x in LOADER_MAPPING.keys()],
        accept_multiple_files=True,
        label_visibility="collapsed",
    ):
        total_docs = len(uploaded_files)
        if st.sidebar.button("Upload"):
            for i, uploaded_file in enumerate(uploaded_files):
                bytes_data = uploaded_file.read()
                with open(f"source_documents/{uploaded_file.name}", "wb") as f:
                    f.write(bytes_data)
                if i == total_docs - 1:
                    pass
                args = FakeArgs()
                ingest_docs(args)

# Add sliders to sidebar
st.sidebar.subheader("Remote Selection")
col1, col2 = st.sidebar.columns(2)
remote_emb = col1.checkbox("Embeddings")
remote_model = col2.checkbox("Model")

st.sidebar.subheader("Model Parameters")
max_tokens = st.sidebar.slider("Max Tokens", 128, 4096, step=128, value=4096)
paragraph_overlap = st.sidebar.slider("Paragraph Overlap", 0, 2, step=1, value=1)

params = {
    "remote_emb": remote_emb,
    "remote_model": remote_model,
    "max_tokens_field": max_tokens,
    "paragraph_overlap": paragraph_overlap,
}

# Main content

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Write your question here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Display assistant response in chat message container
    assistant = st.chat_message("assistant")
    with st.spinner("Thinking..."):
        full_response = inference(
            {"query": prompt, "params": params},
            callbacks=[SimpleStreamlitCallbackHandler(message_area=assistant)],
        )

    if assistant:
        with st.expander("Document Similarity Search"):
            # Find the relevant pages
            # Write out the first
            st.write(full_response["docs"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response["answer"],
            "docs": full_response["docs"],
        }
    )

col1, col2 = st.sidebar.columns(2)


if clear_chat := col1.button("Clear chat"):
    st.session_state.messages = []

if save_chat := col2.button("Save chat"):
    with open(f"logs/chat_history_{datetime.now()}.json", "w") as f:
        json.dump(st.session_state.messages, f)
