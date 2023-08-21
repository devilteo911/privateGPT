import re
import pandas as pd
import requests
from datetime import datetime
from typing import List
from tqdm.auto import tqdm


def load_texts_from_bundle(path: str, mode: str = "texts") -> List[str]:
    """
    Loads the texts from a ResourceBundle file and returns a list of strings
    containing the sentences.

    Args:
        path (str): The path to the ResourceBundle file.

    Returns:
        List[str]: A list of strings containing the texts from the ResourceBundle file.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines: List[str] = f.readlines()
    texts: List[str] = []
    for line in lines:
        if "=" in line:
            if mode == "texts":
                text: str = line.split("=")[-1].strip().replace("\\n", " ")
                texts.append(text)
            else:  # I take only the keys
                text: str = line.split("=")[0].strip().replace("\\n", " ")
                texts.append(text)
    return texts


def main(args):
    df = pd.DataFrame(columns=["question", "answer"])
    texts = load_texts_from_bundle(
        "resources/ResourceBundle_it.properties_1680512691959"
    )

    template = """
    USER: Devi crearmi venti domande il più plausibili possibile per il seguente testo.
    Le venti domande devono attenersi fedelmente al testo. Ecco il testo: {}
    ASSISTANT:
    """
    for text in tqdm(texts):
        q = template.format(text)

        data = {
            "query": q,
            "params": args["params"],
            "callbacks": [],
        }

        response = requests.post(args["url"], json=data)

        if response.status_code == 200:
            ans = response.json()["generations"][0][0]["text"]
            ans = [
                re.sub(r"^\d+\. ", "", s.strip()) for s in ans.split("\n") if s.strip()
            ]
            df = df.append({"question": text, "answer": ans}, ignore_index=True)

        else:
            print(response.status_code)
    df.to_json(args["filename"], orient="records", force_ascii=False)


if __name__ == "__main__":
    url = "http://localhost:7700/api/simple_gen"
    params = {"temperature": 0.1, "top_p": 0.2}
    filename = f"logs/output_{datetime.now()}.json"
    kwargs = {"url": url, "params": params, "filename": filename}
    main(kwargs)
