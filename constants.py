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
