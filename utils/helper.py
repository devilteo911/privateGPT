from datetime import datetime
import json
import pandas as pd


class QALogger:
    def __init__(self, params):
        self.params = params
        self.df: pd.DataFrame = pd.DataFrame(columns=["Question", "Answer"])
        self.filename = f"logs/{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}.json"

    def add_row(self, row_data):
        self.df.loc[len(self.df)] = row_data

    def save_to_disk(self):
        qa_json = self.df.to_dict(orient="records")
        self.params["qa"] = qa_json
        with open(self.filename, "w") as f:
            json.dump(self.params, f, ensure_ascii=False)