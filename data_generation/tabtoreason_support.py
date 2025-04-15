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
import os


nest_asyncio.apply()  # Allows nested event loops (only needed for Jupyter)
tqdm.pandas()

# load dataset

data = pd.read_csv("../tabular_datasets/support_data.csv")

# remove irrelevant columns

data = data[[x for x in data.columns if "Unnamed: 0" not in x]].reset_index(drop=True)

column_names = list(data.columns)

# generate dictionary for mapping column names to meaningful clinical names

# column_mapping = prompt_model(CONVERT_COLUMNS_PROMPT.format(column_names = column_names))

# print(column_mapping["answer"])

renamed_data = data.rename(columns=DICTIONARY_TO_CLINICAL_NAMES_SUPPORT)

trace_data = renamed_data.drop_duplicates(
    subset=[
        x
        for x in renamed_data.columns
        if x != "Patient Death Indicator (0=Alive, 1=Deceased)"
        and x != "Time to Death (Days post-admission)"
    ]
).reset_index(drop=True)

y = trace_data["Patient Death Indicator (0=Alive, 1=Deceased)"]

X = trace_data[
    [
        x
        for x in trace_data.columns
        if x != "Patient Death Indicator (0=Alive, 1=Deceased)"
        and x != "Time to Death (Days post-admission)"
    ]
]

# generate distance matrix

# distances = gower.gower_matrix(X)
# np.save("distances.npy", distances)
# distances = np.load("distances.npy")

# should save the matrix


dictionaries = X.apply(lambda x: dict(x), axis=1)

patient_description_prompts = [
    f"{FROM_JSON_TO_QUESTION_PROMPT_SUPPORT}{json_representation}"
    for json_representation in dictionaries
]

if __name__ == "__main__":
    # batching currently done with 1000 batch size
    def ceiling_division(n, d):
        return -(n // -d)

    batch_size = 1000

    for iteration in range(
        ceiling_division(len(patient_description_prompts), batch_size)
    ):
        print("Running batch:", iteration)
        file_path = f"data/support_data/patient_descriptions/patient_descriptions_support_batch_{iteration}.pkl"
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
