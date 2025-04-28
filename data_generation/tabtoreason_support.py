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

data = data[
    [x for x in data.columns if "Unnamed: 0" not in x and "adls" not in x]
].reset_index(drop=True)


def collapse_income_columns(df, income_columns):
    """
    Collapse multiple income range columns into a single column.

    Args:
        df: pandas DataFrame
        income_columns: list of column names representing income ranges

    Returns:
        A pandas Series with the income ranges
    """
    # Create a temporary dataframe with just income columns
    temp_df = df[income_columns]

    # For each row, get the column name where value is True/1
    income_series = temp_df.idxmax(axis=1)

    # Set to NA where no income column is True/1
    income_series[temp_df.sum(axis=1) == 0] = pd.NA

    return income_series


# Usage:
income_cols = ["under $11k", "$11-$25k", "$25-$50k", ">$50k"]
data["income"] = collapse_income_columns(data, income_cols)

race_cols = ["white", "black", "asian", "hispanic"]
data["race"] = collapse_income_columns(data, race_cols)

data = data[
    [x for x in data.columns if x not in income_cols and x not in race_cols]
].reset_index(drop=True)


# column_names = list(data.columns)

# generate dictionary for mapping column names to meaningful clinical names

# column_mapping = prompt_model(CONVERT_COLUMNS_PROMPT.format(column_names=column_names))

# print(column_mapping["answer"])

renamed_data = data.rename(columns=DICTIONARY_TO_CLINICAL_NAMES_SUPPORT)

trace_data = renamed_data.drop_duplicates(
    subset=[
        x
        for x in renamed_data.columns
        if x != "Mortality Status" and x != "Time to Death (days)"
    ]
).reset_index(drop=True)

y = trace_data["Mortality Status"]

X = trace_data[
    [
        x
        for x in trace_data.columns
        if x != "Mortality Status" and x != "Time to Death (days)"
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

    file_path = f"data/support_data/patient_descriptions/patient_descriptions_support_batch_complete.pkl"

    results = asyncio.run(
        run_llm_calls(patient_description_prompts)
    )  # Run tasks properly

    with open(file_path, "wb") as f:
        pkl.dump(results, f)

    print("Collected", len(results), "responses.")
    # break
