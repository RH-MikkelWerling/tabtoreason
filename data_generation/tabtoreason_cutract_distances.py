# %% Load the data
import pandas as pd
import gower
import numpy as np

cutract_processed_data = pd.read_csv("data/cutract/cutract_no_one_hot.csv")

display(cutract_processed_data)
cutract_processed_data.columns
# %% calculate the distances
event_column = "mort"
time_to_event_column = "days"


X = cutract_processed_data[
    [
        x
        for x in cutract_processed_data.columns
        if x != event_column
        and x != time_to_event_column
        and x != "patient_description_reasoning"
        and x != "patient_description_answering"
    ]
]

y = cutract_processed_data[event_column]

# generate distance matrix

distances = gower.gower_matrix(X)
np.save("data/cutract/processed_data/distances.npy", distances)
# %%
distances = np.load("data/cutract/processed_data/distances.npy")
distances
# %%
