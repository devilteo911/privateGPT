import streamlit as st
from ingest import LOADER_MAPPING
from routes.overloadPDF import inference
from utils.utils import SimpleStreamlitCallbackHandler

# Sidebar
st.sidebar.title("Menu")

# Add sliders to sidebar
remote_emb = st.sidebar.checkbox("Remote Embeddings")
remote_model = st.sidebar.checkbox("Remote Model")
max_tokens = st.sidebar.slider("Max Tokens", 128, 4096, step=128)
paragraph_overlap = st.sidebar.selectbox("Paragraph Overlap", [0, 1, 2])

params = {
    "remote_emb": remote_emb,
    "remote_model": remote_model,
    "max_tokens_field": max_tokens,
    "paragraph_overlap": paragraph_overlap,
}

# Main content
st.title("⚡ Overload PDF Chat 🤖")

uploaded_files = st.file_uploader(
    "Upload a file",
    type=[x.replace(".", "") for x in LOADER_MAPPING.keys()],
    accept_multiple_files=True,
)

total_docs = len(uploaded_files)
if st.button("Upload"):
    for i, uploaded_file in enumerate(uploaded_files):
        bytes_data = uploaded_file.read()
        with open(f"source_documents/{uploaded_file.name}", "wb") as f:
            f.write(bytes_data)
        # if i == total_docs - 1:
        #     pass
        # args = FakeArgs()
        # ingest_docs(args)

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
