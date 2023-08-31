from datetime import datetime
import json
import os
from typing import Any, List
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


def remove_duplicated_paragraphs(
    results: List[List[dict[str:Any]]],
) -> List[dict[str:Any]]:
    """
    Remove duplicated paragraphs from a list of dictionaries.

    Args:
        results (List[List[dict[str:Any]]]): A list of lists of dictionaries. Each
            dictionary represents a paragraph and contains the keys "doc_id" and
            "source".

    Returns:
        List[dict[str:Any]]: A list of dictionaries representing unique paragraphs.
            Each dictionary contains the keys "doc_id" and "source".

    """
    unique_dicts = {}

    # Iterate over each list of dictionaries
    for dictionary_list in results:
        for dictionary in dictionary_list:
            # Get the values of "doc_id" and "source" keys
            doc_id = dictionary["doc_id"]
            source = dictionary["source"]

            # Generate a composite key using tuple
            composite_key = (doc_id, source)

            # Check if the composite key exists in the dictionary
            if composite_key not in unique_dicts:
                # Add the dictionary to the dictionary with composite key
                unique_dicts[composite_key] = dictionary

    # Extract the unique dictionaries from the dictionary with composite key
    unique_dicts = list(unique_dicts.values())

    return unique_dicts
