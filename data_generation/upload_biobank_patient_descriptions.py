from constants import *
import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset
import pickle as pkl

data = pd.read_csv("data/biobank/processed_data/biobank_complete_full_batch.csv")

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

medqa_training_questions = load_dataset("HPAI-BSC/MedQA-Mixtral-CoT", trust_remote_code=True)

medqa_training_questions = pd.DataFrame(medqa_training_questions["train"])
#remove one weird training question
medqa_training_questions = medqa_training_questions[medqa_training_questions["question"] != medqa_training_questions["question"][6]].reset_index(drop = True)

medqa_training_questions = medqa_training_questions.rename(columns = {"question": "problem", "response": "solution"})[["problem", "solution"]].reset_index(drop = True)

def wrap_reasoning_with_tokens(response_text):
    # Remove trailing whitespace and split into lines
    stripped_text = response_text.rstrip()
    lines = stripped_text.split('\n')
    
    if not lines:
        return ""
    
    # Extract answer (last line) and reasoning (all preceding lines)
    answer_line = lines[-1]
    reasoning_text = '\n'.join(lines[:-1])
    
    # Wrap reasoning with <think> tags and append the answer
    formatted_output = f"<think>\n{reasoning_text}\n</think>\n{answer_line}"
    return formatted_output

medqa_training_questions["solution"] = medqa_training_questions["solution"].apply(wrap_reasoning_with_tokens)

# notebook_login()

data = data.reset_index().rename(
    columns={
        "prompts": "problem",
        "patient_description_answering": "answer",
        "patient_description_reasoning": "reasoning",
        "concatenated_reasoning_and_answers": "solution",
    }
)[["problem", "answer", "reasoning", "solution"]]

data = pd.concat([data, medqa_training_questions])

data = data.reset_index().rename(columns = {"index":"id"})

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

# NOTE: This ONLY makes sense if the distances were first calculated AFTER SPLITTING, so not doing it currently

dataset_split = dataset["train"].train_test_split(test_size = 0.05, seed=42)

dataset_split.push_to_hub(
    "mikkel-werling/cardiovascular_biobank_patient_descriptions", private=True
)

# Load the dataset directly from the Hub
dataset = load_dataset("mikkel-werling/cardiovascular_biobank_patient_descriptions")

data = pd.read_csv("data/biobank/processed_data/biobank_complete_full_batch.csv")

file_path = (
        f"data/biobank/tasks/patient_tasks_counterfactuals_biobank_batch_complete.pkl"
    )

with open(file_path, "rb") as f:
    task_responses = pkl.load(f)


file_path = (
        f"data/biobank/processed_data/patient_tasks_counterfactuals_biobank_batch_complete.pkl"
    )

with open("data/biobank/processed_data/biobank_tasks.pkl", "rb") as f:
    task_descriptions = pkl.load(f)

answers = [x["answer"] for x in task_responses]
reasons = [x["reasoning"] for x in task_responses]

data["task_answers"] = answers
data["task_reasoning"] = reasons 

data["prompts"] = task_descriptions

data["concatenated_reasoning_and_answers"] = (
    data["task_reasoning"] + data["task_answers"]
)

# notebook_login()

data = data.reset_index().rename(
    columns={
        "prompts": "problem",
        "patient_description_answering": "answer",
        "patient_description_reasoning": "reasoning",
        "concatenated_reasoning_and_answers": "solution",
    }
)[["problem", "answer", "reasoning", "solution"]]

data = pd.concat([data, medqa_training_questions])

data = data.reset_index().rename(columns = {"index":"id"})


def get_messages_column_from_standard_format(row):
    list_of_messages = [
        {"content": row["problem"], "role": "user"},
        {"content": row["solution"], "role": "assistant"},
    ]

    return list_of_messages


data["messages"] = data.apply(
    lambda x: get_messages_column_from_standard_format(x), axis=1
)

# NOTE: This ONLY makes sense if the distances were first calculated AFTER SPLITTING, so not doing it currently

dataset = DatasetDict({"train": Dataset.from_pandas(data)})

dataset_split = dataset["train"].train_test_split(test_size = 0.05, seed=42)

dataset_split.push_to_hub(
    "mikkel-werling/cardiovascular_biobank_tasks", private=True
)



ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/sft.py \
    --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/sft/config_cardiovascular_biobank_patient_descriptions.yaml


ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/sft.py \
    --config recipes/DeepSeek-R1-Distill-Qwen-7B/sft/config_cardiovascular_biobank_patient_descriptions.yaml

