import requests
import streamlit as st

st.set_page_config(page_title="Overload PDF Chat", layout="wide")

# Sidebar
st.sidebar.title("Menu")

# Add sliders to sidebar
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5, 0.1)
top_k = st.sidebar.slider("Top K", 1, 100, 10, 1)
top_p = st.sidebar.slider("Top P", 0.0, 1.0, 0.9, 0.1)
repeat_penalty = st.sidebar.slider("Repeat Penalty", 1.0, 2.0, 1.2, 0.1)


params = {
    "temperature": temperature,
    "top_k": top_k,
    "top_p": top_p,
    "repeat_penalty": repeat_penalty,
}


# Main content"
st.title("⚡ Overload PDF Chat 🤖")
query = st.text_input("Enter your question here")
if st.button("Get Answer"):
    response = requests.post(
        "http://localhost:7700/api/overloadPDF",
        json={"query": query, "params": params},
        stream=True,
    )
    if response.status_code == 200:
        res = response.json()
        st.write(res["answer"])
        with st.expander("Document Similarity Search"):
            # Find the relevant pages
            # Write out the first
            st.write(res["docs"])
    else:
        st.write("Failed to get answer")
