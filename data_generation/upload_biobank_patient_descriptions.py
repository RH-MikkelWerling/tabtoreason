from constants import *
import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset

data = pd.read_csv("data/biobank/processed_data/biobank_complete.csv")


X = data[
    [
        x
        for x in data.columns
        if x != "Clinical Event Occurrence"
        and x != "Time to Clinical Event (Days)"
        and x != "index"
        and x != "patient_description_reasoning"
        and x != "patient_description_answering"
    ]
]

y = data["Clinical Event Occurrence"]

dictionaries = X.apply(lambda x: dict(x), axis=1)

patient_description_prompts = [
    f"{FROM_JSON_TO_QUESTION_PROMPT_BIOBANK}{json_representation}"
    for json_representation in dictionaries
]

data["prompts"] = patient_description_prompts

data["concatenated_reasoning_and_answers"] = (
    data["patient_description_reasoning"] + data["patient_description_answering"]
)

# notebook_login()

data = data.reset_index().rename(
    columns={
        "index": "id",
        "prompts": "problem",
        "patient_description_answering": "answer",
        "patient_description_reasoning": "reasoning",
        "concatenated_reasoning_and_answers": "solution",
    }
)[["id", "problem", "answer", "reasoning", "solution"]]


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

dataset_split = dataset["train"].train_test_split(test_size = 0.1, seed=42)

dataset_split.push_to_hub(
    "mikkel-werling/cardiovascular_biobank_patient_descriptions", private=True
)

from datasets import load_dataset

# Load the dataset directly from the Hub
dataset = load_dataset("mikkel-werling/cardiovascular_biobank_patient_descriptions")
print(dataset["train"])  # Should show ~100,000 examples
dataset

# dataset_test = load_dataset(
#     "mikkel-werling/cardiovascular_biobank_patient_descriptions", "default"
# )
0.15.4

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/sft.py \
    --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/sft/config_cardiovascular_biobank_patient_descriptions.yaml


ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/sft.py \
    --config recipes/DeepSeek-R1-Distill-Qwen-7B/sft/config_cardiovascular_biobank_patient_descriptions.yaml
