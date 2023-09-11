#!/usr/bin/env python3
import os
import sys

from dotenv import load_dotenv
from fastapi import APIRouter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pydantic import BaseModel
import weaviate

from utils.helper import QALogger
from constants import PARAMS, QUESTIONS, QUESTIONS_MULTI_DOC
from utils.utils import (
    SimpleStreamlitCallbackHandler,
    load_llm_and_retriever,
    overwrite_llm_params,
    parse_arguments,
    retrieve_document_neighborhood,
    select_retrieval_chain,
)


class Query(BaseModel):
    query: str
    params: dict


load_dotenv()
router = APIRouter()
args = parse_arguments()
params = PARAMS.copy()


@router.post("/simple_gen")
def simple_gen(query: Query):
    params.update(query.params)

    ggml_model, _ = load_llm_and_retriever(
        params,
        callbacks=[StreamingStdOutCallbackHandler(), SimpleStreamlitCallbackHandler()],
        rest=True,
    )
    ggml_model = overwrite_llm_params(ggml_model, params)

    gen = ggml_model.generate(prompts=[query.query])
    return gen


def inference(query: Query, callbacks):
    params.update(query["params"])

    ggml_model, db_client = load_llm_and_retriever(params, callbacks, rest=True)
    ggml_model = overwrite_llm_params(ggml_model, params)
    qa = select_retrieval_chain(ggml_model, params)

    # Interactive questions and answers
    query = query["query"]

    relevant_docs = retrieve_document_neighborhood(db_client, query, params)

    # Get the answer from the chain
    res = qa({"input_documents": relevant_docs, "question": query})
    answer, docs = (
        res["output_text"],
        [] if args.hide_source else res["input_documents"],
    )

    docs_to_return = []

    # # Print the relevant sources used for the answer
    for document in docs:
        docs_to_return.append(document.page_content)

    return {"answer": answer, "docs": docs_to_return}


@router.post("/multiTest")
def multi_test(
    query: Query, questions=None, callbacks=[StreamingStdOutCallbackHandler()]
):
    params.update(query.params)
    ggml_model, retriever = load_llm_and_retriever(params, callbacks, rest=True)
    llm = overwrite_llm_params(ggml_model, params)
    qa = select_retrieval_chain(llm, retriever, params)

    logs = QALogger(params)
    docs_to_return = []
    if not questions:
        questions = QUESTIONS + QUESTIONS_MULTI_DOC
    # Interactive questions and answers
    while True:
        try:
            if len(questions) == 0:
                logs.save_to_disk()
                break
            query = questions[0]
            print("\nAuto Query: ", query)
            questions.pop(0)

            # Get the answer from the chain
            res = qa(query)
            answer, docs = (
                res["result"],
                [] if args.hide_source else res["source_documents"],
            )

            # writing the results on file
            logs.add_row((query, answer))
            logs.save_to_disk()

            # Print the relevant sources used for the answer
            for document in docs:
                #     print("\n> " + document.metadata["source"] + ":")
                #     print(document.page_content)
                docs_to_return.append(document.page_content)
        except KeyboardInterrupt:
            logs.save_to_disk()
            sys.exit()
    return "DONE"


@router.post("/simpleChat")
def simple_chat(query: dict, callbacks=[StreamingStdOutCallbackHandler()]):
    params.update(query["params"])
    ggml_model, _ = load_llm_and_retriever(params, callbacks, rest=True)
    llm = overwrite_llm_params(ggml_model, params)
    out = llm.generate(prompts=[query["query"]]).generations[0][0].text
    return out
