import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset

data = pd.read_csv("data/biobank/response_data.csv")

# notebook_login()

data = data.reset_index().rename(
    columns={
        "index": "id",
        "task": "problem",
        "answers_task": "answer",
        "reasoning_task": "solution",
    }
)[["id", "problem", "answer", "solution"]]


def get_messages_column_from_standard_format(row):
    list_of_messages = [
        {"content": row["problem"], "role": "user"},
        {"content": row["solution"], "role": "assistant"},
    ]

    return list_of_messages


data["messages"] = data.apply(
    lambda x: get_messages_column_from_standard_format(x), axis=1
)

dataset = DatasetDict({"train": Dataset.from_pandas(data)})

dataset.push_to_hub("mikkel-werling/cardiovascular_biobank")

dataset_test = load_dataset("mikkel-werling/cardiovascular_biobank")

dataset_test

dataset_test = load_dataset("open-r1/OpenR1-Math-220k")

dataset_test

print(dataset_test["train"].map())

# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
#     src/open_r1/sft.py \
#     --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/sft/config_cardiovascular_biobank.yaml
