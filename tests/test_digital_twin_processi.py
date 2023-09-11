import os
import sys
import pymongo
import requests
from tqdm.auto import tqdm
from dotenv import load_dotenv

abspath = os.path.abspath(".")
sys.path.append(abspath)

from constants import DIGITAL_TWIN_TEMPLATE, DT_SYSTEM_TEMPLATE, QUESTIONS_DT, COT
from utils.helper import QALogger

load_dotenv()

url = "http://localhost:7701/api/simpleChat"


def get_data_from_pymongo():
    client = pymongo.MongoClient(
        host=os.environ["DB_HOSTNAME"],
        authSource=os.environ["DB_AUTH_SOURCE"],
        authMechanism="SCRAM-SHA-1",
    )
    db = client[os.environ["DB_AUTH_SOURCE"]]
    collection = db["sequences"]
    data = collection.find()
    return list(data)


def main():
    params = {
        "temperature": 0.0,
        "max_tokens": 4096,
    }
    logger = QALogger(params)
    db_data = list(get_data_from_pymongo())

    for question in tqdm(QUESTIONS_DT):
        query = DIGITAL_TWIN_TEMPLATE.format(db_data, COT, question)
        data = {"query": query, "params": params}
        response = requests.post(url, json=data)
        logger.add_row((question, response.text))
    logger.params["system_prompt"] = DT_SYSTEM_TEMPLATE
    logger.save_to_disk()


if __name__ == "__main__":
    main()
