import os
import sys
import pymongo
import requests
from tqdm.auto import tqdm
from dotenv import load_dotenv
import json
from bson.objectid import ObjectId

abspath = os.path.abspath(".")
sys.path.append(abspath)

from constants import DIGITAL_TWIN_TEMPLATE, DT_PERIOD_SYSTEM_TEMPLATE, DT_SYSTEM_TEMPLATE, QUESTION_PERIOD_DT, QUESTIONS_DT, COT
from utils.helper import QALogger

load_dotenv()

url = "http://localhost:7701/api/simpleChat"

map_pd_days_2_days = {
    "B": "processo giornaliero",
    "W": "processo settimanale",
    "M": "processo mensile",
    "3M": "processo trimestrale",
    "A": "processo annuale",
    "B+": "processo eseguito più volte al giorno",
    "W+": "processo eseguito più volte a settimana",
    "M+": "processo eseguito più volte al mese",
    "3M+": "processo eseguito più volte al trimestre",
    "A+": "processo eseguito più volte all'anno",
}


def get_data_from_pymongo():
    client = pymongo.MongoClient(
        host=os.environ["DB_HOSTNAME"],
        authSource=os.environ["DB_AUTH_SOURCE"],
        authMechanism="SCRAM-SHA-1",
    )
    db = client[os.environ["DB_AUTH_SOURCE"]]
    collection = db["periodic"]
    data = collection.find()
    return list(data)

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def pp_data(data):
    data_filtered = []
    for d in data:
        for x in ["_id", "company", "time_range", "count"]:
            d.pop(x)
        d = flatten_dict(d)
        d["period"] = d.pop("period.str")
        d["period"] = map_pd_days_2_days[d["period"]]
        data_filtered.append(d)
    return data_filtered


def convert_objectid(obj):
    if isinstance(obj, ObjectId):
        return str(obj)
    return obj


def main():
    params = {
        "temperature": 0.0,
        "max_tokens": 4096,
    }
    logger = QALogger(params)
    db_data = list(get_data_from_pymongo())

    db_data = pp_data(db_data)

    db_data = json.dumps(db_data, default=convert_objectid)

    for question in tqdm(QUESTION_PERIOD_DT):
        query = DIGITAL_TWIN_TEMPLATE.format(db_data, COT, question)
        data = {"query": query, "params": params}
        response = requests.post(url, json=data)
        logger.add_row((question, response.text))
    logger.params["system_prompt"] = DT_PERIOD_SYSTEM_TEMPLATE
    logger.params["cot"] = COT
    logger.save_to_disk()


if __name__ == "__main__":
    main()
