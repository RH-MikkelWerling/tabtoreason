import pandas as pd
import gower
import numpy as np
from constants import *

data = pd.read_csv("data/covid/covid_no_one_hot.csv")

data = data[[x for x in data.columns if "Unnamed: 0" not in x]].reset_index(drop=True)

column_names = list(data.columns)

# generate dictionary for mapping column names to meaningful clinical names

# column_mapping = prompt_model(CONVERT_COLUMNS_PROMPT.format(column_names = column_names))

# print(column_mapping["answer"])
# %%
renamed_data = data.rename(columns=DICTIONARY_TO_CLINICAL_NAMES_COVID)

trace_data = renamed_data.drop_duplicates(
    subset=[
        x
        for x in renamed_data.columns
        if x != "Deceased Status" and x != "Days from Hospital Admission to Outcome"
    ]
).reset_index(drop=True)

X = trace_data[
    [
        x
        for x in trace_data.columns
        if x != "Deceased Status"
        and x != "Days from Hospital Admission to Outcome"
    ]
]

y = trace_data["Deceased Status"]

# generate distance matrix

distances = gower.gower_matrix(X)
np.save("data/covid/processed_data/distances.npy", distances)
# distances = np.load("distances.npy")

# %%
