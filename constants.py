import os
from dotenv import load_dotenv

load_dotenv()

QUESTIONS = []

QUESTIONS_MULTI_DOC = []

STUFF_TEMPLATE = """### System : You are an AI assistant that helps people find information using the context provided to answer the question of the User. 
If the answer is not in the context just say that you don't know in a polite way, don't try to make up an answer. 
The text you will find in the context will have all the information you need to answer the question. 
You MUST always include the file, the chapter and the page in which you found the information!
You MUST provide an helpful answer in syntactically correct Italian!

Context:
---------
{context}
---------

### USER: {question}.

### ASSISTANT:"""


PARAMS = {
    "embedding_model": os.environ.get("EMBEDDINGS_MODEL_NAME"),
    "target_source_chunks": int(os.environ.get("TARGET_SOURCE_CHUNKS", 4)),
    "qa": [],
    "temperature": 0.0,
    "max_tokens_field": 512,
    "paragraph_overlap": 0,
    "top_k": 50,
    "top_p": 0.9,
    "repeat_penalty": 1.2,
    "chain_type": "stuff",
    "remote_emb": False,
    "remote_model": False,
}
