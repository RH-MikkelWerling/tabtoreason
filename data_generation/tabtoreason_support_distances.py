# %% Load the data
import pandas as pd
import gower
import numpy as np

support_processed_data = pd.read_csv("data/support_data/support_data.csv",index_col=0)

support_processed_data
support_processed_data.columns
# %% calculate the distances
event_column = "death"
time_to_event_column = "d.time"


X = support_processed_data[
    [
        x
        for x in support_processed_data.columns
        if x != event_column
        and x != time_to_event_column
        and x != "patient_description_reasoning"
        and x != "patient_description_answering"
    ]
]

y = support_processed_data[event_column]

# generate distance matrix

distances = gower.gower_matrix(X)
np.save("data/support_data/processed_data/distances.npy", distances)
# %%
distances = np.load("data/support_data/processed_data/distances.npy")
distances