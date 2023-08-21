import argparse
import os
import shutil
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import langchain
import streamlit as st
import weaviate
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
from langchain import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceInstructEmbeddings, OpenAIEmbeddings
from langchain.llms import CTransformers, GPT4All, LlamaCpp, OpenAI
from langchain.llms.base import LLM
from langchain.output_parsers import RegexParser
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever
from langchain.schema import Document
from langchain.vectorstores import Weaviate
from langchain.vectorstores.base import VectorStore
from loguru import logger

from constants import (
    CHROMA_SETTINGS,
    COMBINED_TEMPLATE,
    PERSIST_DIRECTORY,
    QUESTION_TEMPLATE,
    STUFF_TEMPLATE,
)
from ingest import does_vectorstore_exist
from ingest import main as ingest_docs

langchain.verbose = True

load_dotenv()
openai_api_key_emb = os.environ.get("OPENAI_API_KEY")
openai_api_key_mock = os.environ.get("OPENAI_API_KEY_MOCK")
openai_api_base_mock = os.environ.get("OPENAI_API_BASE_MOCK")


@dataclass
class FakeArgs:
    chunk_size: int
    chunk_overlap: int
    debug: bool = False
    rest: bool = False


class SimpleStreamlitCallbackHandler(BaseCallbackHandler):
    """Callback Handler that logs to streamlit."""

    def __init__(self, message_area) -> None:
        self.tokens_area = message_area.empty()
        self.tokens_stream = ""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        with self.tokens_area:
            self.tokens_stream += token
            self.tokens_area.markdown(self.tokens_stream + "▌")


def initialize_llm(params, callbacks, rest=False):
    """
    Initializes a language model (LLM) based on the given parameters.

    Args:
        params (dict): A dictionary containing the parameters for initializing the LLM.
        callbacks (list): A list of callback functions to be used during LLM initialization.

    Returns:
        langchain.llms.base.LLM: The initialized LLM object.
    """
    if rest:
        openai_params = {
            "temperature": params["temperature"],
            "top_p": params["top_p"],
            "max_tokens": params["max_tokens_field"],
            "streaming": True,
            "callbacks": callbacks,
        }
        if params["remote_model"]:
            openai_params["model_name"] = "gpt-3.5-turbo"
            openai_params.pop("top_p")
            return ChatOpenAI(**openai_params)
        else:
            # this call a local model that can output in chatgpt format
            openai_params["openai_api_key"] = openai_api_key_mock
            openai_params["openai_api_base"] = openai_api_base_mock
            return OpenAI(**openai_params)
    else:
        # Prepare the LLM
        match params["model_type"]:
            case "LlamaCpp":
                llm = LlamaCpp(
                    model_path=params["model_path"],
                    n_ctx=params["model_n_ctx"],
                    callbacks=callbacks,
                    verbose=False,
                    temperature=params["temperature"],
                    top_k=params["top_k"],
                    top_p=params["top_p"],
                    repeat_penalty=params["repeat_penalty"],
                    n_gpu_layers=2000000,
                    n_batch=512,
                    n_threads=8,
                    streaming=True,
                )

            case "CTransformers":
                config = {
                    "gpu_layers": 40,
                    "temperature": params["temperature"],
                    "max_new_tokens": 1024,
                    "stream": True,
                }
                llm = CTransformers(
                    model=params["model_path"], config=config, model_type="mpt"
                )
            case "GPT4All":
                llm = GPT4All(
                    model=params["model_path"],
                    n_ctx=params["model_n_ctx"],
                    backend="gptj",
                    callbacks=callbacks,
                    verbose=False,
                )
            case _default:
                print(f"Model {params['model_type']} not supported!")
                exit

    return llm


def overwrite_llm_params(llm: LLM, params: Dict[str, float]) -> LLM:
    """
    Overwrites the temperature, top_k, top_p, and repeat_penalty parameters of the given LLM object with the values
    specified in the params dictionary.

    Args:
        llm (langchain.llms.base.LLM): The LLM object to modify.
        params (dict): A dictionary containing the new parameter values.

    Returns:
        langchain.llms.base.LLM: The modified LLM object.
    """
    # logger.info(f"Before: {llm.temperature, llm.top_p}")
    llm.temperature = params["temperature"]
    # llm.top_p = params["top_p"]
    # logger.info(f"After: {llm.temperature, llm.top_p}")

    return llm


def load_llm_and_retriever(
    params: Dict[str, any], callbacks, rest=False
) -> Tuple[LLM, Weaviate]:
    """
    Loads a language model and a retriever based on the given parameters.

    Args:
        params (dict): A dictionary containing the parameters for loading the model and retriever.

    Returns:
        tuple: A tuple containing the loaded language model and retriever.
    """
    # Parse the command line arguments

    logger.info(f"Current params: {params}")

    model_kwargs = {"device": "cuda:1"}
    if params["remote_emb"]:
        logger.info("Using OpenAI embeddings")
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key_emb)
    else:
        logger.info("Using HuggingFace embeddings")
        encode_kwargs = {"normalize_embeddings": True}
        embeddings = HuggingFaceInstructEmbeddings(
            model_name=params["embedding_model"],
            model_kwargs=model_kwargs,
            query_instruction="Represent this sentence for searching relevant passages:",
            encode_kwargs=encode_kwargs,
        )

    db_client = weaviate.Client(url=os.environ["WEAVIATE_URL"])

    db = Weaviate(
        client=db_client,
        index_name="Overload_chat",
        embedding=embeddings,
        text_key="text",
        by_text=False,
    )

    llm = initialize_llm(params, callbacks, rest=rest)

    return llm, db


def select_retrieval_chain(llm: LLM, retriever: VectorStore, params: dict):
    """
    Selects a retrieval chain based on the given chain type and returns a RetrievalQA object.

    Args:
        llm (langchain.llms.base.LLM): The LLM object to use for the retrieval chain.
        retriever (langchain.vectorstores.base.VectorStore): The vector store to use for the retrieval chain.
        params (dict): A dictionary containing the parameters for the retrieval chain.

    Returns:
        langchain.chains.RetrievalQA: A RetrievalQA object based on the selected retrieval chain.
    """
    match params["chain_type"]:
        case "map_reduce":
            QUESTION_PROMPT = PromptTemplate(
                template=QUESTION_TEMPLATE, input_variables=["context", "question"]
            )
            COMBINE_PROMPT = PromptTemplate(
                template=COMBINED_TEMPLATE, input_variables=["summaries", "question"]
            )
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type=params["chain_type"],
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={
                    "question_prompt": QUESTION_PROMPT,
                    "combine_prompt": COMBINE_PROMPT,
                },
            )
        case "stuff_old":
            output_parser = RegexParser(
                regex=r"(.*?)\nScore: (.*)",
                output_keys=["answer", "score"],
            )

            STUFF_PROMPT = PromptTemplate(
                template=STUFF_TEMPLATE,
                input_variables=["context", "question"],
                output_parser=output_parser,
            )

            chain_type_kwargs = {"prompt": STUFF_PROMPT}
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type=params["chain_type"],
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs=chain_type_kwargs,
            )
        case "stuff":
            output_parser = RegexParser(
                regex=r"(.*?)\nScore: (.*)",
                output_keys=["answer", "score"],
            )
            STUFF_PROMPT = PromptTemplate(
                template=STUFF_TEMPLATE,
                input_variables=["context", "question"],
                output_parser=output_parser,
            )
            qa = load_qa_chain(llm, prompt=STUFF_PROMPT, chain_type="stuff")

        case _default:
            print(f"Chain type {params['chain_type']} not supported!")
            exit
    return qa


def check_stored_embeddings(params: dict):
    """
    Checks if stored embeddings exist and updates them if necessary. If we choose to use
    remote embeddings, we delete the local embeddings and update them with the remote
    ones and vice versa.

    Args:
        params (dict): A dictionary containing parameters for the embeddings.

    Returns:
        None
    """
    logger.info(f"{'REMOTE' if params['remote_emb'] else 'LOCAL'} embeddings selected.")
    emb_type = "remote" if params["remote_emb"] else "local"
    emb_saved_type = f"db/{emb_type}_emb.dummy"
    if not os.path.exists(emb_saved_type) or not does_vectorstore_exist(
        persist_directory=PERSIST_DIRECTORY
    ):
        shutil.rmtree("db", ignore_errors=True)
        chunk_size = 1500 if params["remote_emb"] else 450
        args = FakeArgs(
            chunk_size=chunk_size,
            chunk_overlap=0,
            debug=False,
            rest=params["remote_emb"],
        )
        ingest_docs(args)
        Path(emb_saved_type).touch()


def retrieve_document_neighborhood(
    retriever: VectorStore, query: str, params: dict
) -> List[Document]:
    """
    Retrieve the neighborhood of documents around the top-k documents returned by a similarity search.

    Args:
        retriever (VectorStore): A VectorStore object used to perform the similarity search.
        query (str): The query string used to perform the similarity search.
        params (dict): A dictionary containing parameters for the neighborhood retrieval.

    Returns:
        List[Document]: A list of dictionaries representing the documents in the neighborhood.
    """
    k = params["target_source_chunks"]
    overlap = params["paragraph_overlap"]

    candidate_docs = retriever.similarity_search(
        query=query, k=k, distance_metric="cos"
    )

    if overlap != 0 or params["remote_emb"]:
        # Picking the neighborhood of the each document based on its id
        # FIXME: the get() method is not present with Weaviate DBs
        getter = retriever.get()
        all_docs_and_metas = {
            k["id"]: v for k, v in zip(getter["metadatas"], getter["documents"])
        }

        candidate_docs_id = [doc.metadata["id"] for doc in candidate_docs]

        # adding the overlap neighborhood ids

        candidate_docs_id = add_prev_next(candidate_docs_id, all_docs_and_metas)

        docs = [
            all_docs_and_metas[k] for k in candidate_docs_id if k in all_docs_and_metas
        ]

        metadatas = [
            next(item for item in getter["metadatas"] if item["id"] == id_)
            for id_ in candidate_docs_id
        ]

        candidate_docs = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(docs, metadatas)
        ]

    return candidate_docs


def add_prev_next(mylist: List[int], all_docs: List[Document]) -> List[int]:
    """
    Given a list of indices `mylist` and a list of all documents `all_docs`, returns a new list
    containing the indices of the previous and next documents for each index in `mylist`. If an index
    is at the beginning or end of `all_docs`, the previous or next index will wrap around to the
    opposite end of the list.

    Args:
        mylist (List[int]): A list of indices.
        all_docs (List[Document]): A list of all documents.

    Returns:
        List[int]: A new list containing the indices of the previous and next documents for each index
        in `mylist`.
    """
    output = []
    for i in mylist:
        if i == 0:
            # Underflow - add next 2
            output.extend([i, i + 1, i + 2])
        elif i == len(all_docs):
            # Overflow - add prev 2
            output.extend([i - 2, i - 1, i])
        else:
            prev = i - 1 if i > 0 else len(all_docs) - 1
            next = i + 1 if i < len(all_docs) - 1 else 0
            output.extend([prev, i, next])

    # FIXME: the list gets reordered, is this something we want?
    return list(set(output))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="OverloadChat: Ask questions to your documents without an internet connection, "
        "using the power of LLMs."
    )
    parser.add_argument(
        "--hide-source",
        "-S",
        action="store_true",
        help="Use this flag to disable printing of source documents used for answers.",
    )

    parser.add_argument(
        "--mute-stream",
        "-M",
        action="store_true",
        help="Use this flag to disable the streaming StdOut callback for LLMs.",
    )

    parser.add_argument(
        "--chat_history",
        "-H",
        action="store_true",
        help="Use this flag to save the log of the chat to the disk",
    )

    parser.add_argument(
        "--automated_test",
        "-A",
        action="store_true",
        help="Use this flag to test a set of question in order to judge the quality of the model",
    )

    return parser.parse_args()
