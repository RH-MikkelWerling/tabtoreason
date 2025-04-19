import pandas as pd
import gower
import numpy as np

biobank_processed_data = pd.read_csv("data/biobank/processed_data/biobank.csv")

biobank_processed_data

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
