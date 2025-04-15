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

# column_mapping = prompt_model(CONVERT_COLUMNS_PROMPT.format(column_names = column_names))

# print(column_mapping["answer"])

renamed_data = data.rename(columns=DICTIONARY_TO_CLINICAL_NAMES_BIOBANK)

renamed_data = renamed_data.drop_duplicates(
    subset=[
        x
        for x in renamed_data.columns
        if x != "Clinical Event Occurrence" and x != "Time to Clinical Event (Days)"
    ]
).reset_index(drop=True)

# so here, what we need to do is to exclude the first 5000 from each class, and then let's get to 50000

renamed_data = renamed_data.reset_index()

renamed_positives = (
    renamed_data[renamed_data["Clinical Event Occurrence"] == 1]
    .head(5000)
    .reset_index(drop=True)
)


renamed_negatives = (
    renamed_data[renamed_data["Clinical Event Occurrence"] == 0]
    .head(5000)
    .reset_index(drop=True)
)

trace_data = pd.concat([renamed_positives, renamed_negatives]).reset_index(drop=True)

all_responses_list = []

for batch_index in range(10):
    with open(
        f"data/biobank/patient_descriptions/patient_descriptions_biobank_batch_{batch_index}.pkl",
        "rb",
    ) as f:
        file = pkl.load(f)
        all_responses_list.extend(file)

reasoning_responses = [x["reasoning"] for x in all_responses_list]
answer_responses = [x["answer"] for x in all_responses_list]

(
    trace_data["patient_description_reasoning"],
    trace_data["patient_description_answering"],
) = (reasoning_responses, answer_responses)

trace_data_first_batches = trace_data.reset_index(drop=True)

already_processed_patients = trace_data_first_batches["index"].values

renamed_data_filtered = renamed_data[
    ~renamed_data["index"].isin(already_processed_patients)
]

positives = renamed_data_filtered[
    renamed_data_filtered["Clinical Event Occurrence"] == 1
].reset_index(drop=True)

n_positives = len(positives)

negatives = (
    renamed_data_filtered[renamed_data_filtered["Clinical Event Occurrence"] == 0]
    .head(40000 - n_positives)
    .reset_index(drop=True)
)

trace_data = pd.concat([positives, negatives]).reset_index(drop=True)

all_responses_list = []

for batch_index in range(10, 50):
    with open(
        f"data/biobank/patient_descriptions/patient_descriptions_biobank_batch_{batch_index}.pkl",
        "rb",
    ) as f:
        file = pkl.load(f)
        all_responses_list.extend(file)

reasoning_responses = [x["reasoning"] for x in all_responses_list]
answer_responses = [x["answer"] for x in all_responses_list]

(
    trace_data["patient_description_reasoning"],
    trace_data["patient_description_answering"],
) = (reasoning_responses, answer_responses)

already_processed_patients = pd.concat(
    [trace_data_first_batches, trace_data]
).reset_index(drop=True)

already_processed_patients_index = already_processed_patients["index"].values

trace_data = renamed_data[
    ~renamed_data["index"].isin(already_processed_patients_index)
].reset_index(drop=True)

all_responses_list = []

for batch_index in range(50, 106):
    with open(
        f"data/biobank/patient_descriptions/patient_descriptions_biobank_batch_{batch_index}.pkl",
        "rb",
    ) as f:
        file = pkl.load(f)
        all_responses_list.extend(file)

reasoning_responses = [x["reasoning"] for x in all_responses_list]
answer_responses = [x["answer"] for x in all_responses_list]

(
    trace_data["patient_description_reasoning"],
    trace_data["patient_description_answering"],
) = (reasoning_responses, answer_responses)

biobank_processed_data = pd.concat(
    [already_processed_patients, trace_data]
).reset_index(drop=True)

biobank_processed_data.to_csv("data/biobank/processed_data/biobank.csv", index=False)


y = trace_data["Clinical Event Occurrence"]

X = trace_data[
    [
        x
        for x in trace_data.columns
        if x != "Clinical Event Occurrence"
        and x != "Time to Clinical Event (Days)"
        and x != "index"
    ]
]

X = biobank_processed_data[
    [
        x
        for x in biobank_processed_data.columns
        if x != "Clinical Event Occurrence"
        and x != "Time to Clinical Event (Days)"
        and x != "index"
        and x != "patient_description_reasoning"
        and x != "patient_description_answering"
    ]
]

y = biobank_processed_data["Clinical Event Occurrence"]

# generate distance matrix

distances = gower.gower_matrix(X)
np.save("data/biobank/processed_data/distances.npy", distances)
# distances = np.load("distances.npy")

# should save the matrix


dictionaries = X.apply(lambda x: dict(x), axis=1)

patient_description_prompts = [
    f"{FROM_JSON_TO_QUESTION_PROMPT_BIOBANK}{json_representation}"
    for json_representation in dictionaries
]

if __name__ == "__main__":
    # batching currently done hardcoded
    def ceiling_division(n, d):
        return -(n // -d)

    batch_size = 1000

    for iteration in range(
        ceiling_division(len(patient_description_prompts), batch_size)
    ):
        print("Running batch:", iteration)
        file_path = f"data/biobank/patient_descriptions/patient_descriptions_biobank_batch_{iteration+50}.pkl"
        if os.path.exists(file_path):
            continue
        else:
            current_patient_descriptions = patient_description_prompts[
                iteration * 1000 : (iteration + 1) * 1000
            ]

            results = asyncio.run(
                run_llm_calls(current_patient_descriptions)
            )  # Run tasks properly

            with open(file_path, "wb") as f:
                pkl.dump(results, f)

            print("Collected", len(results), "responses.")
            # break
