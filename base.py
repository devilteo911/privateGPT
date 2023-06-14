import json
from datetime import datetime
from typing import List

import pandas as pd
from langchain.embeddings.base import Embeddings
from transformers import AutoModel, AutoTokenizer


class T5Embedder(Embeddings):
    def __init__(self, model_name, ctx_len: int = 512) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.ctx_len = ctx_len

    @staticmethod
    def average_pool(last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        query_texts = ["passage: " + text for text in texts]
        batch_dict = self.tokenizer(
            query_texts,
            max_length=self.ctx_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        outputs = self.model(**batch_dict)
        return self.average_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        query_texts = ["query: " + text]
        batch_dict = self.tokenizer(
            query_texts,
            max_length=self.ctx_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        outputs = self.model(**batch_dict)
        return self.average_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )[0].tolist()

    def get_tokenizer(self) -> AutoTokenizer:
        return self.tokenizer


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
            json.dump(self.params, f)
