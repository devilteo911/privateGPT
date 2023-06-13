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

QUESTIONS = [
    "che documenti devo presentare per il rimborso?",
    "come faccio a richiedere la prestazione?",
    "quanto pago in una struttura non convenzionata",
    "qaunto pago per i ticket?",
    "che massimali ho?",
    "che modalita' di prestazioni sono previste?",
    "Quanto pago per in caso di utilizzo dei ticket?",
    "Quali sono le strutture convenzionate con il Network Previmedical per effettuare le prestazioni in forma diretta?",
]

QUESTIONS_MULTI_DOC = [
    "Quali sono i massimali di tutte le polizze?",
    "Qual è il massimale per la polizza base p?",
    "Qual è la quota a carico dell'assicurazione?",
]

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
