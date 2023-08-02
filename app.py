import streamlit as st
from ingest import LOADER_MAPPING
from routes.overloadPDF import inference
from utils.utils import SimpleStreamlitCallbackHandler

# Sidebar
st.sidebar.title("Menu")

# Add sliders to sidebar
remote_emb = st.sidebar.checkbox("Remote Embeddings")
remote_model = st.sidebar.checkbox("Remote Model")
max_tokens = st.sidebar.slider("Max Tokens", 128, 4096, 128)

params = {
    "remote_emb": remote_emb,
    "remote_model": remote_model,
    "max_tokens_field": max_tokens,
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

query = st.text_input("Enter your question here")
if st.button("Get Answer"):
    with st.spinner("typing..."):
        res = inference(
            {"query": query, "params": params},
            callbacks=[SimpleStreamlitCallbackHandler()],
        )

    with st.expander("Document Similarity Search"):
        # Find the relevant pages
        # Write out the first
        st.write(res["docs"])
