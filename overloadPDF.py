#!/usr/bin/env python3
import sys
from typing import List
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp, CTransformers
from base import Logs, T5Embedder
from constants import COMBINED_TEMPLATE, QUESTION_TEMPLATE, QUESTIONS
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

import os
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path


from methods import pick_logs_filename
from constants import CHROMA_SETTINGS

load_dotenv()


persist_directory = os.environ.get("PERSIST_DIRECTORY")

params = {
    "model_path": os.environ.get("MODEL_PATH"),
    "model_type": os.environ.get("MODEL_TYPE"),
    "embedding_model": os.environ.get("EMBEDDINGS_MODEL_NAME"),
    "model_n_ctx": os.environ.get("MODEL_N_CTX"),
    "target_source_chunks": int(os.environ.get("TARGET_SOURCE_CHUNKS", 4)),
    "qa": [],
    "temperature": 0.2,
    "top_k": 50,
    "top_p": 0.2,
    "repeat_penalty": 1.2,
    "chain_type": "stuff",
}


def choose_model(params, callbacks):
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


def main():
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceInstructEmbeddings(model_name=params["embedding_model"])
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
    )
    retriever = db.as_retriever(search_kwargs={"k": params["target_source_chunks"]})
    # activate/deactivate the streaming StdOut callback for LLMs

    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    llm = choose_model(params, callbacks)

    # QUESTION_PROMPT = PromptTemplate(
    #     template=QUESTION_TEMPLATE, input_variables=["context", "question"]
    # )
    # COMBINE_PROMPT = PromptTemplate(
    #     template=COMBINED_TEMPLATE, input_variables=["summaries", "question"]
    # )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=params["chain_type"],
        retriever=retriever,
        return_source_documents=not args.hide_source,
        # chain_type_kwargs={
        #     "question_prompt": QUESTION_PROMPT,
        #     "combine_prompt": COMBINE_PROMPT,
        # },
    )

    logs = Logs(params)
    # Interactive questions and answers
    while True:
        try:
            if not args.automated_test:
                query = input("\nEnter a query: ")
                if query == "exit":
                    break
            else:
                if len(QUESTIONS) == 0:
                    logs.save_to_disk()
                    break
                query = QUESTIONS[0]
                print("Auto Query: ", query)
                QUESTIONS.pop(0)

            # Get the answer from the chain
            res = qa(query + ". Answer in italian.")
            answer, docs = (
                res["result"],
                [] if args.hide_source else res["source_documents"],
            )

            # writing the results on file

            logs.add_row((query, answer))

            # Print the result
            print("\n\n> Question:")
            print(query)
            print("\n> Answer:")
            print(answer)

            # Print the relevant sources used for the answer
            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)
        except KeyboardInterrupt:
            logs.save_to_disk()
            sys.exit()


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
        action="store_false",
        help="Use this flag to test a set of question in order to judge the quality of the model",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
