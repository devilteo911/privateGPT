from typing import List
from langchain.embeddings.base import Embeddings
from transformers import AutoTokenizer, AutoModel

class T5Embedder(Embeddings):
    def __init__(self, model_name) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
    
    @staticmethod
    def average_pool(last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        query_texts = ["passage: " + text for text in texts]
        batch_dict = self.tokenizer(query_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
        outputs = self.model(**batch_dict)
        return self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask']).tolist()

    def embed_query(self, text: str) -> List[float]:
        query_texts = ["query: " + text]
        batch_dict = self.tokenizer(query_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
        outputs = self.model(**batch_dict)
        return self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])[0].tolist()