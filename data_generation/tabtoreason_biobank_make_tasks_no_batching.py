from constants import *
from prompt_functions import *
import pandas as pd
import numpy as np
from tqdm import tqdm
import asyncio
import nest_asyncio  # Fix for Jupyter Notebooks
import pickle as pkl

nest_asyncio.apply()  # Allows nested event loops (only needed for Jupyter)
from tqdm.asyncio import tqdm
import gower
import seaborn as sns


tqdm.pandas()

# load dataset

trace_data = pd.read_csv("data/biobank/processed_data/biobank_complete.csv")


y = trace_data["Clinical Event Occurrence"]

X = trace_data[
    [
        x
        for x in trace_data.columns
        if x != "Clinical Event Occurrence" and x != "Time to Clinical Event (Days)"
    ]
]

# get distances
# distances = gower.gower_matrix(X)
# np.save("distances.npy", distances)
distances = np.load("data/biobank/processed_data/distances.npy")

response_data = trace_data.rename(
    columns={
        "patient_description_reasoning": "reasoning",
        "patient_description_answering": "answers",
        "Clinical Event Occurrence": "outcome",
    }
).reset_index(drop=True)


def find_closest_by_label(distance_matrix, labels):
    num_samples = distance_matrix.shape[0]
    closest_samples = np.zeros(
        (num_samples, 2), dtype=int
    )  # Store closest [label_0, label_1] indices
    closest_distances = np.zeros(
        (num_samples, 2), dtype=float
    )  # Store min distances [label_0, label_1]

    for i in tqdm(range(num_samples)):
        # Exclude self-distance
        distances = distance_matrix[i].copy()
        distances[i] = np.inf  # Ignore the self-distance

        # Find closest for label 0
        mask_0 = labels == 0
        valid_distances_0 = np.where(mask_0, distances, np.inf)
        closest_0 = np.argmin(valid_distances_0)
        min_dist_0 = valid_distances_0[closest_0]

        # Find closest for label 1
        mask_1 = labels == 1
        valid_distances_1 = np.where(mask_1, distances, np.inf)
        closest_1 = np.argmin(valid_distances_1)
        min_dist_1 = valid_distances_1[closest_1]

        # Store results
        closest_samples[i] = [closest_0, closest_1]
        closest_distances[i] = [min_dist_0, min_dist_1]

    return closest_samples, closest_distances


closest_samples, closest_distances = find_closest_by_label(distances, y.values)

closest_sample_0 = [x[0] for x in closest_samples]

closest_sample_1 = [x[1] for x in closest_samples]

closest_distance_0 = [x[0] for x in closest_distances]

closest_distance_1 = [x[1] for x in closest_distances]

response_data["closest_sample_0"] = closest_sample_0
response_data["closest_sample_1"] = closest_sample_1

response_data["closest_distance_0"] = closest_distance_0
response_data["closest_distance_1"] = closest_distance_1


def make_label_specific_columns(row):
    if row["outcome"] == 0:
        same_label_closest_distance = row["closest_distance_0"]
        other_label_closest_distance = row["closest_distance_1"]
    else:
        same_label_closest_distance = row["closest_distance_1"]
        other_label_closest_distance = row["closest_distance_0"]
    return same_label_closest_distance, other_label_closest_distance


response_data[["closest_distance_same_label", "closest_distance_other_label"]] = (
    response_data.apply(
        lambda x: make_label_specific_columns(x), axis=1, result_type="expand"
    )
)


def calculate_difficulty_of_task(row, alpha=0.5):
    difficulty = (
        alpha * row["closest_distance_other_label"]
        + (1 - alpha) * row["closest_distance_same_label"]
    )
    return difficulty


def calculate_difficulty_of_task(row):
    difficulty = (row["closest_distance_same_label"]) / (
        row["closest_distance_other_label"]
    )
    return 1 / (1 + np.exp(-difficulty))


response_data["difficulty"] = response_data.apply(
    lambda x: calculate_difficulty_of_task(x), axis=1
)

# answer_responses

response_data["outcome_text"] = response_data["outcome"].apply(
    lambda x: "Dead" if x == 1 else "Alive"
)

response_data["answers"] = response_data["answers"].apply(lambda x: x.strip())

# sns.displot(data=response_data, x="difficulty")


def generate_tasks_from_patient_descriptions(row, response_data, system_prompt):
    target_outcome, target_description = (
        row["outcome_text"],
        row["answers"],
    )
    survivor_description = response_data["answers"][row["closest_sample_0"]]
    survivor_outcome = "Alive"
    death_description = response_data["answers"][row["closest_sample_1"]]
    death_outcome = "Dead"

    task_prompt = system_prompt.format(
        survivor_description=survivor_description,
        death_description=death_description,
        target_description=target_description,
        target_outcome=target_outcome,
    )
    return task_prompt


tasks = response_data.apply(
    lambda x: generate_tasks_from_patient_descriptions(
        x, response_data, FROM_PATIENT_DESCRIPTIONS_TO_TASK_PROMPT
    ),
    axis=1,
).to_list()

# all_responses_list = []

# for batch_index in range(10):
#     with open(
#         f"data/biobank/tasks/patient_tasks_biobank_batch_{batch_index}.pkl", "rb"
#     ) as f:
#         file = pkl.load(f)
#         all_responses_list.extend(file)

# reasoning_responses = [x["reasoning"] for x in all_responses_list]
# answer_responses = [x["answer"] for x in all_responses_list]

# response_data["reasoning_task"] = reasoning_responses
# response_data["answers_task"] = answer_responses
# response_data["task"] = tasks

# response_data["reasoning_task_length"] = response_data["reasoning_task"].apply(
#     lambda x: len(x)
# )

# sns.displot(data=response_data, x="reasoning_task_length")

# sns.lmplot(data=response_data, x="difficulty", y="reasoning_task_length", scatter=True)

# file
# Example usage
if __name__ == "__main__":
    import os

    file_path = (
        f"data/biobank/tasks/patient_tasks_counterfactuals_biobank_batch_complete.pkl"
    )

    results = asyncio.run(run_llm_calls(tasks))  # Run tasks properly

    with open(file_path, "wb") as f:
        pkl.dump(results, f)

    print("Collected", len(results), "responses.")
