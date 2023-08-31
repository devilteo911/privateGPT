import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import langchain
import weaviate
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.llms import CTransformers, GPT4All, LlamaCpp, OpenAI
from langchain.llms.base import LLM
from langchain.output_parsers import RegexParser
from langchain.schema import Document
from langchain.vectorstores import Weaviate
from loguru import logger

from constants import (
    STUFF_TEMPLATE,
)
from utils.helper import remove_duplicated_paragraphs

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

    # TODO: once we finish dealing with local embedding with weaviate we get back to this
    # if params["remote_emb"]:
    #     logger.info("Using OpenAI embeddings")
    #     embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key_emb)

    db = weaviate.Client(url=os.environ["WEAVIATE_URL"])
    llm = initialize_llm(params, callbacks, rest=rest)

    return llm, db


def select_retrieval_chain(llm: LLM, params: dict):
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

        case "map_reduce":
            raise NotImplementedError
        case _default:
            print(f"Chain type {params['chain_type']} not supported!")
            exit
    return qa


def retrieve_document_neighborhood(
    retriever: weaviate.Client, query: str, params: dict
) -> List[Document]:
    """
    Retrieve the neighborhood of documents around the top-k documents returned by a similarity search.

    Args:
        retriever (weaviate.Client): A VectorStore object used to perform the similarity search.
        query (str): The query string used to perform the similarity search.
        params (dict): A dictionary containing parameters for the neighborhood retrieval.

    Returns:
        List[Document]: A list of dictionaries representing the documents in the neighborhood.
    """

    overlap = params["paragraph_overlap"]
    k = (
        params["target_source_chunks"]
        if overlap > 0
        else params["target_source_chunks"] + 4
    )
    class_name = os.environ["WEAVIATE_INDEX_NAME"]

    candidate_docs = (
        retriever.query.get(class_name, ["text", "source", "doc_id", "page"])
        .with_hybrid(query, properties=["text"])
        .with_limit(k)
        .with_additional(["score"])
        .do()
    )["data"]["Get"][class_name]

    if overlap != 0 or params["remote_emb"]:
        # Picking the neighborhood of the each document based on its id
        candidate_docs_id = [{doc["source"]: doc["doc_id"]} for doc in candidate_docs]

        # adding the overlap neighborhood ids
        candidate_paragraph = add_prev_next(retriever, candidate_docs_id)

        metadatas = [
            {"source": doc["source"], "page": doc["page"], "doc_id": doc["doc_id"]}
            for doc in candidate_paragraph
        ]

        candidate_docs = [
            Document(page_content=doc["text"], metadata=meta)
            for doc, meta in zip(candidate_paragraph, metadatas)
        ]
    else:
        metadatas = [
            {"source": doc["source"], "page": doc["page"], "doc_id": doc["doc_id"]}
            for doc in candidate_docs
        ]
        candidate_docs = [
            Document(page_content=doc["text"], metadata=meta)
            for doc, meta in zip(candidate_docs, metadatas)
        ]

    return candidate_docs


def add_prev_next(retriever: weaviate.Client, candidates: dict):
    index_name = os.environ["WEAVIATE_INDEX_NAME"]
    result = []
    for elem in candidates:
        source = next(iter(elem.keys()))
        id_ = elem[source]
        if id_ == 0:
            lower_limit = id_ - 1
            upper_limit = id_ + 3
        else:
            lower_limit = id_ - 2
            upper_limit = id_ + 2

        query = (
            retriever.query.get(index_name, ["doc_id", "source", "page", "text"])
            .with_limit(1000)
            .with_where(
                {
                    "operator": "And",
                    "operands": [
                        {
                            "path": ["source"],
                            "operator": "Equal",
                            "valueText": source,
                        },
                        {
                            "operator": "And",
                            "operands": [
                                {
                                    "path": ["doc_id"],
                                    "operator": "GreaterThan",
                                    "valueNumber": lower_limit,
                                },
                                {
                                    "path": ["doc_id"],
                                    "operator": "LessThan",
                                    "valueNumber": upper_limit,
                                },
                            ],
                        },
                    ],
                }
            )
        )

        result.append(query.do()["data"]["Get"][index_name])
    result = remove_duplicated_paragraphs(result)
    return result


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
