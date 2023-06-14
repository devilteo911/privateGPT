#!/usr/bin/env python3
import sys
from dotenv import load_dotenv
from pydantic import BaseModel
from base import QALogger
from constants import QUESTIONS


from constants import PARAMS
from fastapi import APIRouter
from utils.utils import (
    load_llm_and_retriever,
    overwrite_llm_params,
    parse_arguments,
    select_retrieval_chain,
)


class Query(BaseModel):
    query: str
    params: dict


load_dotenv()
router = APIRouter()
args = parse_arguments()
params = PARAMS.copy()
ggml_model, retriever = load_llm_and_retriever(params, rest=True)


# @router.post("/overloadPDF")
def inference(query: Query):
    params.update(query["params"])

    # llm = overwrite_llm_params(ggml_model, params)
    llm = ggml_model
    qa = select_retrieval_chain(llm, retriever, params)

    docs_to_return = []
    # Interactive questions and answers
    while True:
        try:
            if not args.automated_test:
                try:
                    query = query["query"]
                except AttributeError:
                    break

            # Get the answer from the chain
            res = qa(query + ". Answer in italian.")
            print(res)
            answer, docs = (
                res["result"],
                [] if args.hide_source else res["source_documents"],
            )

            # # Print the relevant sources used for the answer
            for document in docs:
                docs_to_return.append(document.page_content)
        except KeyboardInterrupt:
            sys.exit()
    return {"answer": answer, "docs": docs_to_return}


@router.post("/multiTest")
def multi_test(query: Query):
    params.update(query.params)

    llm = overwrite_llm_params(ggml_model, params)
    qa = select_retrieval_chain(llm, retriever, params)

    logs = QALogger(params)
    docs_to_return = []
    # Interactive questions and answers
    while True:
        try:
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
            logs.save_to_disk()

            # Print the relevant sources used for the answer
            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)
                docs_to_return.append(document.page_content)
        except KeyboardInterrupt:
            logs.save_to_disk()
            sys.exit()
    return "DONE"