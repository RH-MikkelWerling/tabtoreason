# %%
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

# %%

nest_asyncio.apply()  # Allows nested event loops (only needed for Jupyter)
tqdm.pandas()

# load dataset

data = pd.read_csv("../tabular_datasets/Maggic.csv")

# %%
# remove irrelevant columns

column_names = list(data.columns)

# generate dictionary for mapping column names to meaningful clinical names

#column_mapping = prompt_model(CONVERT_COLUMNS_PROMPT.format(column_names = column_names))

# print(column_mapping["answer"])
# %%
renamed_data = data.rename(columns=DICTIONARY_TO_CLINICAL_NAMES_MAGGIC)

trace_data = renamed_data.drop_duplicates(
    subset=[
        x
        for x in renamed_data.columns
        if x != "Deceased Status" and x != "Days from Hospital Admission to Outcome"
    ]
).reset_index(drop=True)

y = trace_data["Deceased Status"]

X = trace_data[
    [
        x
        for x in trace_data.columns
        if x != "Deceased Status" and x != "Days from Hospital Admission to Outcome"
    ]
]

# generate distance matrix

# distances = gower.gower_matrix(X)
# np.save("distances.npy", distances)
# distances = np.load("distances.npy")

# should save the matrix


dictionaries = X.apply(lambda x: dict(x), axis=1)

patient_description_prompts = [
    f"{FROM_JSON_TO_QUESTION_PROMPT_COVID}{json_representation}"
    for json_representation in dictionaries
]
# %%
if __name__ == "__main__":
    # batching currently done with 1000 batch size
        file_path = f"data/covid/patient_descriptions/patient_descriptions_covid_complete.pkl"

        results = asyncio.run(
            run_llm_calls(patient_description_prompts)
        )  # Run tasks properly

        with open(file_path, "wb") as f:
            pkl.dump(results, f)

        print("Collected", len(results), "responses.")
        # break