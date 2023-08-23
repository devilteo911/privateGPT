from datetime import datetime
import json
import os
from typing import List
import pandas as pd
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()


class QALogger:
    def __init__(self, params):
        self.params = params
        self.df: pd.DataFrame = pd.DataFrame(columns=["Question", "Answer"])
        self.filename = f"logs/{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}.json"

    def add_row(self, row_data):
        self.df.loc[len(self.df)] = row_data

    def save_to_disk(self):
        qa_json = self.df.to_dict(orient="records")
        self.params["qa"] = qa_json
        with open(self.filename, "w") as f:
            json.dump(self.params, f, ensure_ascii=False)


def get_all_documents_from_db(db_client) -> List[Document]:
    """
    This is a workaround for the fact that Weaviate does not provide a get method.
    Retrieve all documents from the database based on a query.
    Args:
        retriever: The retriever object used to perform the search.
        query: The query string used to search for documents.

    Returns:
        A list of Document objects that match the given query.
    """
    index_name = os.environ["WEAVIATE_INDEX_NAME"]
    results = db_client.query.get(
        index_name, ["doc_id", "source", "page", "text"]
    ).do()["data"]["Get"][index_name]
    return {
        "documents": [x.pop("text") for x in results],
        "metadatas": [x for x in results],
    }
