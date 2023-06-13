#!/usr/bin/env python3
import sys
from typing import List
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp, CTransformers
from base import T5Embedder
from constants import QUESTIONS

import os
import argparse
from datetime import datetime
from pathlib import Path

from methods import pick_logs_filename

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get("PERSIST_DIRECTORY")

model_type = os.environ.get("MODEL_TYPE")
model_path = os.environ.get("MODEL_PATH")
model_n_ctx = os.environ.get("MODEL_N_CTX")
target_source_chunks = int(os.environ.get("TARGET_SOURCE_CHUNKS", 4))

from constants import CHROMA_SETTINGS


def main():
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = T5Embedder(model_name=embeddings_model_name)
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
    )
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    logs_filename = pick_logs_filename(
        save_path="logs",
        model_path=model_path,
        embeddings_model_name=embeddings_model_name,
        model_n_ctx=model_n_ctx,
        target_source_chunks=target_source_chunks,
    )
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(
                model_path=model_path,
                n_ctx=model_n_ctx,
                callbacks=callbacks,
                verbose=False,
                temperature=0.0,
                n_gpu_layers=2000000,
                n_batch=512,
                n_threads=8,
                streaming=True,
            )

        case "CTransformers":
            config = {
                "gpu_layers": 40,
                "temperature": 0.0,
                "max_new_tokens": 1024,
                "stream": True,
                "threads": 0,
            }
            llm = CTransformers(
                model=model_path,
                model_type="llama",
                config=config,
            )
        case "GPT4All":
            llm = GPT4All(
                model=model_path,
                n_ctx=model_n_ctx,
                backend="gptj",
                callbacks=callbacks,
                verbose=False,
            )
        case _default:
            print(f"Model {model_type} not supported!")
            exit
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=not args.hide_source,
    )

    Path(logs_filename).touch()
    # Interactive questions and answers
    while True:
        try:
            if not args.automated_test:
                query = input("\nEnter a query: ")
                if query == "exit":
                    break
            else:
                if len(QUESTIONS) == 0:
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

            with open(logs_filename, "a") as chat_history:
                chat_history.write("\n\n> Question:")
                chat_history.write(query)
                chat_history.write("\n> Answer:")
                chat_history.write(answer)

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
            chat_history.close()
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
