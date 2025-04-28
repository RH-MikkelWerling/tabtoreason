from constants import *
from prompt_functions import *
import pandas as pd
from litellm import completion, acompletion

# from ucimlrepo import fetch_ucirepo, list_available_datasets
from tqdm import tqdm
import asyncio
import litellm
import nest_asyncio  # Fix for Jupyter Notebooks
import pickle as pkl
import httpx
from tqdm.asyncio import tqdm
import gower
import random
import numpy as np
import os


nest_asyncio.apply()  # Allows nested event loops (only needed for Jupyter)
tqdm.pandas()

# load dataset

data = pd.read_excel("../tabular_datasets/biobank_cvd.xlsx")

# remove irrelevant columns

data = data[[x for x in data.columns if "AUROC" not in x]].reset_index(drop=True)


column_names = list(data.columns)

# generate dictionary for mapping column names to meaningful clinical names

# column_mapping = prompt_model(CONVERT_COLUMNS_PROMPT.format(column_names=column_names))

# print(column_mapping["answer"])

renamed_data = data.rename(columns=DICTIONARY_TO_CLINICAL_NAMES_BIOBANK)

renamed_data = renamed_data.drop_duplicates(
    subset=[
        x
        for x in renamed_data.columns
        if x != "Clinical Event Occurrence" and x != "Time to Clinical Event (Days)"
    ]
).reset_index(drop=True)


y = renamed_data["Clinical Event Occurrence"]

X = renamed_data[
    [
        x
        for x in renamed_data.columns
        if x != "Clinical Event Occurrence"
        and x != "Time to Clinical Event (Days)"
        and x != "index"
    ]
]

file_path = (
    f"data/biobank/patient_descriptions/patient_descriptions_biobank_batch_complete.pkl"
)

with open(file_path, "rb") as f:
    responses = pkl.load(f)

answers = [x["answer"] for x in responses]
reasoning = [x["reasoning"] for x in responses]


# generate distance matrix

# distances = gower.gower_matrix(X)
# np.save("data/biobank/processed_data/distances.npy", distances)
# distances = np.load("distances.npy")

dictionaries = X.apply(lambda x: dict(x), axis=1)

patient_description_prompts = [
    f"{FROM_JSON_TO_QUESTION_PROMPT_BIOBANK}{json_representation}"
    for json_representation in dictionaries
]

columns = {
    "patient_description_reasoning": "reasoning",
    "patient_description_answering": "answers",
    "Clinical Event Occurrence": "outcome",
}

renamed_data["patient_description_reasoning"] = reasoning
renamed_data["patient_description_answering"] = answers

renamed_data.to_csv(
    "data/biobank/processed_data/biobank_complete_full_batch.csv", index=False
)

if __name__ == "__main__":

    file_path = f"data/biobank/patient_descriptions/patient_descriptions_biobank_batch_complete.pkl"

    results = asyncio.run(
        run_llm_calls(patient_description_prompts)
    )  # Run tasks properly

    with open(file_path, "wb") as f:
        pkl.dump(results, f)

    print("Collected", len(results), "responses.")
    # break
