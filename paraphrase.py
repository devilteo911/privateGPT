from datetime import datetime
import json
import re
import pandas as pd
import requests
from tqdm.auto import tqdm
from test import load_texts_from_bundle


def main(args):
    template = """
    USER: I will give you an italian text: I want you to return me 2 paraphrasis of it.
    You must reply in italian. Here it is the italian text: {}
    ASSISTANT:
    """
    keys = load_texts_from_bundle(
        "resources/ResourceBundle_it.properties_1680512691959", mode="keys"
    )

    # load the json from the resources folder
    with open("logs/domande_jurij.json", "r") as f:
        data = json.load(f)

    df = pd.DataFrame(columns=["keys", "question", "answer", "paraphrases"])

    for i, d in tqdm(enumerate(data)):
        _, answers = d["question"], d["answer"]
        paraphrases = []
        for answer in tqdm(answers):
            q = template.format(answer)

            data = {
                "query": q,
                "params": args["params"],
                "callbacks": [],
            }

            response = requests.post(args["url"], json=data)
            if response.status_code == 200:
                ans = response.json()["generations"][0][0]["text"]
                ans = [
                    re.sub(r"^\d+\. ", "", s.strip())
                    for s in ans.split("\n")
                    if s.strip()
                ]
                paraphrases += ans

            else:
                print(response.status_code)
        d["keys"] = keys[i]
        d["paraphrases"] = paraphrases

        # reoreder the keys of the dict
        d = {
            "keys": d["keys"],
            "question": d["question"],
            "answer": d["answer"],
            "paraphrases": d["paraphrases"],
        }

        df = df.append(d, ignore_index=True)

    df.to_json(args["filename"], orient="records", force_ascii=False)


if __name__ == "__main__":
    url = "http://localhost:7700/api/simple_gen"
    params = {"temperature": 0.1, "top_p": 0.2}
    filename = f"logs/jurij_par_{datetime.now()}.json"
    kwargs = {"url": url, "params": params, "filename": filename}
    main(kwargs)
