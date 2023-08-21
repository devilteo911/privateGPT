from datetime import datetime
import json
from typing import List, Dict
from loguru import logger
import streamlit as st
from routes.overloadPDF import inference, simple_chat
from utils.utils import FakeArgs, SimpleStreamlitCallbackHandler

# Sidebar


def inference_with_locked_chat(
    message_history: List[Dict[str, str]], message_area, params
) -> str:
    prompt = """
    ### System: This is a frozen chat between a user and you as its assistant. You must answer the user's questions based on the chat history.
    In order to answer the user's questions, you must only follow the chat history and not use any other information. You must provide
    an helpful answer in a grammatically correct italian. Also you must report correctly any digits, date and number present in the history.

    History:
    ---------
    {}
    ---------

    ### USER: {}

    ### ASSISTANT:"""

    new_context = ""
    last_question_docs = []
    # the button can be pressed even on a user message

    # get the index where the last document is available
    last_index = 2 * (len(message_history[1::2]) - 1) + 1
    last_question_docs = message_history[last_index].get("docs")

    # we pick the elements of the list that share the same documents
    new_context += "\n\n\n".join(last_question_docs) + "\n\n"
    for i in range(1, len(message_history)):
        prev_message = message_history[i - 1]
        curr_message = message_history[i]
        if last_question_docs == curr_message.get("docs"):
            new_context += (
                f"{prev_message.get('role')}: {prev_message.get('content')}\n\n"
            )
            new_context += (
                f"{curr_message.get('role')}: {curr_message.get('content')}\n\n"
            )

    prompt = prompt.format(new_context, message_history[-1].get("content"))
    logger.info(prompt)
    return simple_chat(
        {"query": prompt, "params": params},
        callbacks=[SimpleStreamlitCallbackHandler(message_area)],
    )


def main():
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

    lock_chat = st.sidebar.checkbox("Lock current chat", value=False)

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
            if not lock_chat:
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
            else:
                logger.success("Chat is now locked")
                full_response = inference_with_locked_chat(
                    message_history=st.session_state.messages,
                    message_area=assistant,
                    params=params,
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

    col1, col2 = st.sidebar.columns(2)

    if clear_chat := col1.button("Clear chat"):
        st.session_state.messages = []
        logger.success("Chat history has been reset")

    if save_chat := col2.button("Save chat"):
        with open(f"logs/chat_history_{datetime.now()}.json", "w") as f:
            json.dump(st.session_state.messages, f)
            logger.success(f"Chat history has been saved as {f.name}")


if __name__ == "__main__":
    main()
