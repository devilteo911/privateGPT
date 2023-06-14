from fastapi import Query
import requests
import streamlit as st
from routes.overloadPDF import inference
from utils.utils import SimpleStreamlitCallbackHandler

# st.set_page_config(page_title="Overload PDF Chat", layout="wide")

# Sidebar
st.sidebar.title("Menu")

# Add sliders to sidebar
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5, 0.1)
top_k = st.sidebar.slider("Top K", 1, 100, 10, 1)
top_p = st.sidebar.slider("Top P", 0.0, 1.0, 0.9, 0.1)
repeat_penalty = st.sidebar.slider("Repeat Penalty", 1.0, 2.0, 1.2, 0.1)


params = {
    "temperature": temperature,
    "top_p": top_p,
}


# Main content"
st.title("⚡ Overload PDF Chat 🤖")
query = st.text_input("Enter your question here")
if st.button("Get Answer"):
    res_box = st.empty()
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
