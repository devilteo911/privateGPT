import os
from dotenv import load_dotenv
from chromadb.config import Settings

load_dotenv()

# Define the folder for storing database
PERSIST_DIRECTORY = os.environ.get("PERSIST_DIRECTORY")

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=PERSIST_DIRECTORY,
    anonymized_telemetry=False,
)

# QUESTIONS = [
#     "che documenti devo presentare per il rimborso?",
#     "come faccio a richiedere la prestazione?",
#     "quanto pago in una struttura non convenzionata",
#     "qaunto pago per i ticket?",
#     "che massimali ho?",
#     "che modalita' di prestazioni sono previste?",
#     "Quanto pago per in caso di utilizzo dei ticket?",
#     "Quali sono le strutture convenzionate con il Network Previmedical per effettuare le prestazioni in forma diretta?",
# ]

# QUESTIONS_MULTI_DOC = [
#     "Quali sono i massimali di tutte le polizze?",
#     "Qual è il massimale per la polizza base p?",
#     "Qual è la quota a carico dell'assicurazione?",
# ]

QUESTIONS = []

QUESTIONS_MULTI_DOC = []

QUESTION_TEMPLATE = """Use the following portion of a long document to see if any of the text is relevant to answer the question. 
Return any relevant text in Italian.
{context}
Question: {question}
Relevant text, if any, in Italian:"""

COMBINED_TEMPLATE = """Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
Respond in Italian.

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER IN ITALIAN:"""

STUFF_TEMPLATE = """### System : You are an AI assistant that helps people find information using the context provided to answer the question of the User. 
If the answer is not in the context just say that you don't know in a polite way, don't try to make up an answer. 
The text you will find in the context will have all the information you need to answer the question. 
You MUST provide an helpful answer in syntactically correct Italian!

Context:
---------
{context}
---------

### USER: {question}.

### ASSISTANT:"""


# STUFF_TEMPLATE = """<|system|>"You are an AI assistant that helps people find information. User will you give you a question. Your task is to answer as faithfully as you can. While answering think step-by- step and justify your answer." You must reply in correct Italian. Here it is the context:{context}</s><|prompter|>{question}</s><|assistant|>"""
# STUFF_TEMPLATE = """<|system|>"You are an AI assistant that helps people find information. User will you give you a question. Your task is to answer as faithfully as you can. While answering think step-by- step and justify your answer." You must reply in correct Italian. Here it is the context:{context}</s><|prompter|>{question}</s><|assistant|>"""

PARAMS = {
    "model_path": os.environ.get("MODEL_PATH"),
    "model_type": os.environ.get("MODEL_TYPE"),
    "embedding_model": os.environ.get("EMBEDDINGS_MODEL_NAME"),
    "model_n_ctx": os.environ.get("MODEL_N_CTX"),
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
