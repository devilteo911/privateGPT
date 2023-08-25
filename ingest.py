#!/usr/bin/env python3
import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List

import weaviate
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.embeddings import HuggingFaceInstructEmbeddings, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger
from tqdm import tqdm

load_dotenv()


# Load environment variables
persist_directory = os.environ.get("PERSIST_DIRECTORY")
source_directory = os.environ.get("SOURCE_DIRECTORY", "source_documents")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
openai_api_base = os.environ.get("OPENAI_API_BASE")
openai_api_key = os.environ.get("OPENAI_API_KEY_MOCK")
openai_api_key_emb = os.environ.get("OPENAI_API_KEY")


# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if "text/html content not found in email" in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"] = "text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [
        file_path for file_path in all_files if file_path not in ignored_files
    ]
    results = []
    with tqdm(
        total=len(filtered_files), desc="Loading new documents", ncols=80
    ) as pbar:
        for file_path in filtered_files:
            docs = load_single_document(file_path)
            results.extend(docs)
            pbar.update()
    return results


def save_to_txt(documents):
    for document in documents:
        txt_filename = "".join(document.metadata["source"].split(".")[:-1]) + ".txt"
        Path(txt_filename).write_text(document.page_content)


def add_metadata(documents, inject_in_the_page_content: bool = False):
    for document in documents:
        polizze = document.page_content.split("\n\n")[0].split("+")
        if inject_in_the_page_content:
            document.metadata["__polizze"] = [polizza for polizza in polizze]
            # document.page_content = "Il documento fa riferimento alle polizze: " + ", ".join([polizza for polizza in polizze]) + "\n\n" + document.page_content
        else:
            document.metadata["polizze"] = [polizza for polizza in polizze]
    return documents


def inject_metadata(texts):
    for text in texts:
        context = ""
        for k, v in text.metadata.items():
            if not k.startswith("__"):
                continue
            context += f"Il documento fa riferimento a {k[2:]}: {','.join(v)}\n\n"
        text.page_content = context + text.page_content
        text.metadata.pop(k)
    return texts


def update_metadata(
    texts: List[Document], md5s: List[Dict[str, str]]
) -> List[Document]:
    """
    Update the metadata of a list of texts.

    Args:
        texts (List[Document]): A list of Langchain Documents.
        md5s (List[Dict[str, str]]): A list of dictionaries containing the md5 values for each source.

    Returns:
        List[Document]: A list of Document objects with updated metadata.

    Raises:
        IndexError: If no md5 value is found for the current source.
    """
    prev_source = texts[0].metadata["source"]
    for i, text in enumerate(texts):
        curr_source = texts[i].metadata["source"]
        if curr_source != prev_source:
            prev_source = curr_source
        text.metadata.update({"doc_id": i})
        text.metadata["md5"] = [
            item[prev_source] for item in md5s if prev_source in item
        ][0]
    return texts


def process_documents(
    embeddings, args, md5s, ignored_files: List[str] = []
) -> List[Document]:
    """
    Load documents and split in chunks
    """
    logger.info(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        logger.info("No new documents to load")
    logger.info(f"Loaded {len(documents)} new documents from {source_directory}")
    if not args.rest:
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            embeddings.client.tokenizer,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            separators=["\n"],
        )
    else:
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    texts = text_splitter.split_documents(documents)

    # Call the function with your desired arguments
    texts = update_metadata(texts, md5s)

    if args.debug and not args.rest:
        model_name = embeddings.model_name
        tok_voc = {v: k for k, v in embeddings.client.tokenizer.vocab.items()}
        dict_tokenization = []
        # Not so pythonic loop but it keeps the Document class otherwise texts[i] becomes a tuple
        for i in range(len(texts)):
            tokenization = " ".join(
                [
                    tok_voc[x]
                    for x in embeddings.client.tokenizer(texts[i].page_content)[
                        "input_ids"
                    ]
                ]
            )
            dict_tokenization.append(
                {f"{i}_{texts[i].metadata['source']}": tokenization}
            )
        with open(f"logs/debug_{model_name.split('/')[-1]}.json", "w") as f:
            json.dump(
                dict_tokenization,
                f,
                ensure_ascii=False,
            )

    # texts = inject_metadata(texts)
    logger.info(
        f"Split into {len(texts)} chunks of text (max. {args.chunk_size} tokens each)"
    )
    return texts


def calculate_md5(file_path):
    md5_list = []
    files = Path(file_path).glob("*")

    for filename in files:
        with open(filename, "rb") as f:
            data = f.read()
            file_hash = hashlib.md5(data).hexdigest()

        md5_list.append({str(filename): file_hash})
    return md5_list


def skip_already_processed_documents(md5_list, db_client):
    # TO BE IMPLEMENTED
    pass


def db_is_already_populated(db_client):
    if db_client.data_object.get()["objects"] != []:
        return True
    else:
        return False


def main(args):
    # Create embeddings
    if not args.rest:
        encode_kwargs = {"normalize_embeddings": True}
        embeddings = HuggingFaceInstructEmbeddings(
            model_name=embeddings_model_name,
            model_kwargs={"device": "cuda:1"},
            encode_kwargs=encode_kwargs,
        )
    else:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key_emb)

    db_client = weaviate.Client(url=os.environ["WEAVIATE_URL"])
    md5_list = calculate_md5(source_directory)

    # ignored_files = (
    #     skip_already_processed_documents(md5_list, db_client)
    #     if db_is_already_populated(db_client)
    #     else []
    # )
    ignored_files = []

    # Update and store locally vectorstore
    logger.info("Creating embeddings, it may take some minutes...")
    texts = process_documents(
        embeddings, args=args, ignored_files=ignored_files, md5s=md5_list
    )
    # use weaviate to create data objects to db

    logger.info(f"Appending to db with class_name {os.environ['WEAVIATE_INDEX_NAME']}")

    for text in texts:
        text = dict(text)
        db_client.data_object.create(
            data_object={"text": text.pop("page_content"), **text["metadata"]},
            class_name=os.environ["WEAVIATE_INDEX_NAME"],
        )

    logger.success(
        "Ingestion complete! You can now run overload-chat to query your documents"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process documents for Overload-Chat ingestion"
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Set to True for debug mode",
    )

    parser.add_argument(
        "--rest",
        "-r",
        action="store_true",
        help="Set to True for REST mode",
    )

    parser.add_argument(
        "--chunk_size",
        "-c",
        type=int,
        default=300,
        help="Set the chunk size for the text splitter",
    )
    parser.add_argument(
        "--chunk_overlap",
        "-o",
        type=int,
        default=0,
        help="Set the chunk overlap for the text splitter",
    )

    args = parser.parse_args()

    debug = args.debug
    main(args)
