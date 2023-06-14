import argparse
import os
from typing import Dict, Tuple

from fastapi.responses import StreamingResponse
from langchain import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import CTransformers, GPT4All, LlamaCpp
from langchain.llms.base import LLM
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStore
from loguru import logger

from constants import CHROMA_SETTINGS, COMBINED_TEMPLATE, QUESTION_TEMPLATE

from .fastapi_utils import astreamer


class MyCustomCallbackHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # your function for streaming eg. send a Server Side Event or update realtime database
        return StreamingResponse(astreamer(token), media_type="text/event-stream")


def initialize_llm(params, callbacks):
    """
    Initializes a language model (LLM) based on the given parameters.

    Args:
        params (dict): A dictionary containing the parameters for initializing the LLM.
        callbacks (list): A list of callback functions to be used during LLM initialization.

    Returns:
        langchain.llms.base.LLM: The initialized LLM object.
    """
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
    logger.info(f"Before: {llm.temperature, llm.top_k, llm.top_p, llm.repeat_penalty}")
    llm.temperature = params["temperature"]
    llm.top_k = params["top_k"]
    llm.top_p = params["top_p"]
    llm.repeat_penalty = params["repeat_penalty"]
    logger.info(f"After: {llm.temperature, llm.top_k, llm.top_p, llm.repeat_penalty}")

    return llm


def load_llm_and_retriever(params: Dict[str, any]) -> Tuple[LLM, Chroma]:
    """
    Loads a language model and a retriever based on the given parameters.

    Args:
        params (dict): A dictionary containing the parameters for loading the model and retriever.

    Returns:
        tuple: A tuple containing the loaded language model and retriever.
    """
    # Parse the command line arguments

    logger.info(f"Current params: {params}")

    embeddings = HuggingFaceInstructEmbeddings(model_name=params["embedding_model"])
    db = Chroma(
        persist_directory=os.environ.get("PERSIST_DIRECTORY"),
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
    )
    retriever = db.as_retriever(search_kwargs={"k": params["target_source_chunks"]})
    callbacks = [MyCustomCallbackHandler()]
    llm = initialize_llm(params, callbacks)

    return llm, retriever


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
        case "stuff":
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type=params["chain_type"],
                retriever=retriever,
                return_source_documents=True,
            )
        case _default:
            print(f"Chain type {params['chain_type']} not supported!")
            exit
    return qa


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="privateGPT: Ask questions to your documents without an internet connection, "
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
