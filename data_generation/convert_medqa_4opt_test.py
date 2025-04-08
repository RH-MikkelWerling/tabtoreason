import pandas as pd
import json
from datasets import load_dataset, DatasetDict, Dataset


with open("../benchmarks/medqa_4opt_test.json", "rb") as f:
    data = json.load(f)

dataframe = pd.DataFrame(list(data.values()))

dataframe = dataframe.rename(columns={"correct_answer": "answer_idx"})

for option in ["A", "B", "C", "D"]:
    dataframe[option] = dataframe.apply(lambda x: x["options"].get(option), axis=1)

dataset = DatasetDict({"train": Dataset.from_pandas(dataframe)})

dataset.push_to_hub("mikkel-werling/medqa_4opt")
