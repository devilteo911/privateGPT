#!/usr/bin/env python3
import sys
from dotenv import load_dotenv
from loguru import logger
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


# @router.post("/overloadPDF")
def inference(query: Query, callbacks):
    params.update(query["params"])

    ggml_model, retriever = load_llm_and_retriever(params, callbacks, rest=True)
    ggml_model = overwrite_llm_params(ggml_model, params)
    qa = select_retrieval_chain(ggml_model, retriever, params)

    docs_to_return = []
    # Interactive questions and answers

    query = query["query"]

    # Get the answer from the chain
    res = qa(query + ". Answer in italian.")
    answer, docs = (
        res["result"],
        [] if args.hide_source else res["source_documents"],
    )

    print(res)

    # # Print the relevant sources used for the answer
    for document in docs:
        docs_to_return.append(document.page_content)

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
