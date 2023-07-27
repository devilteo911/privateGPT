from pathlib import Path
import shutil
from fastapi import Query
import requests
import streamlit as st
from ingest import LOADER_MAPPING
from routes.overloadPDF import inference
from utils.utils import FakeArgs, SimpleStreamlitCallbackHandler
from io import StringIO
import sys


# st.set_page_config(page_title="Overload PDF Chat", layout="wide")


# Sidebar
st.sidebar.title("Menu")

# Add sliders to sidebar

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5, 0.1)
top_k = st.sidebar.slider("Top K", 1, 100, 10, 1)
top_p = st.sidebar.slider("Top P", 0.0, 1.0, 0.9, 0.1)
repeat_penalty = st.sidebar.slider("Repeat Penalty", 1.0, 2.0, 1.2, 0.1)
remote_emb = st.sidebar.checkbox("Remote Embeddings")
remote_model = st.sidebar.checkbox("Remote Model")


params = {
    "temperature": temperature,
    "top_p": top_p,
    "remote_emb": remote_emb,
    "remote_model": remote_model,
}

# Main content"
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

    # st.write(res["answer"])
    with st.expander("Document Similarity Search"):
        # Find the relevant pages
        # Write out the first
        st.write(res["docs"])
