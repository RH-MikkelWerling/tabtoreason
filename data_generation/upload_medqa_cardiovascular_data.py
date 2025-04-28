import pandas as pd
import json
from datasets import load_dataset, DatasetDict, Dataset

# from lighteval.tasks.default_tasks

med_qa = load_dataset("bigbio/med_qa", "med_qa_en_source", trust_remote_code=True)

data = pd.read_csv("data/benchmarks/medqa_4opt_with_labels.csv")

dictionary = eval(data["options"].values[0])

items = list(dictionary.items())

correct_format = [{"key": x[0], "value": x[1]} for x in items]


def convert_options_to_medqa_format(option):
    dictionary = eval(option)

    items = list(dictionary.items())

    correct_format = [{"key": x[0], "value": x[1]} for x in items]

    return correct_format


data["options"] = data["options"].apply(lambda x: convert_options_to_medqa_format(x))

cardiovascular_data = data[data["category"] == "Cardiovascular"].reset_index(drop=True)

print(cardiovascular_data["question"].values[0])

dataset = DatasetDict({"test": Dataset.from_pandas(data)})

dataset.push_to_hub("mikkel-werling/medqa_4opt_test")

cardiovascular_dataset = DatasetDict({"test": Dataset.from_pandas(cardiovascular_data)})

cardiovascular_dataset.push_to_hub("mikkel-werling/medqa_4opt_cardiovascular_test")

dataset_test = load_dataset("mikkel-werling/medqa_cardiovascular")
