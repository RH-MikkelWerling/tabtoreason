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
data = pd.read_csv("data/covid/covid_no_one_hot.csv")
# %%
# remove irrelevant columns

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

y = trace_data["Deceased Status"]

with open(
    "data/covid/patient_descriptions/patient_descriptions_covid_complete.pkl", "rb"
) as f:
    patient_descriptions = pkl.load(f)

reasoning = [x["reasoning"] for x in patient_descriptions]
answering = [x["answer"] for x in patient_descriptions]

trace_data["patient_description_reasoning"] = reasoning
trace_data["patient_description_answering"] = answering

# get distances
# distances = gower.gower_matrix(X)
# np.save("distances.npy", distances)
distances = np.load("data/covid/processed_data/distances.npy")

response_data = trace_data.rename(
    columns={
        "patient_description_reasoning": "reasoning",
        "patient_description_answering": "answers",
        "Deceased Status": "outcome",
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


class AdaptiveConcurrencyController:
    def __init__(self, total_requests):
        self.total = total_requests
        self.completed = 0
        self.errors = 0
        self.current_concurrency = 1000  # Start high

    async def adjust(self):
        progress = self.completed / self.total

        if progress < 0.2:  # First 20%
            pass  # Maintain burst
        elif progress < 0.5:  # Next 30%
            self.current_concurrency = 500
        else:  # Final 50%
            if self.errors / self.completed > 0.05:  # >5% error rate
                self.current_concurrency = max(50, self.current_concurrency * 0.9)
            else:
                self.current_concurrency = min(200, self.current_concurrency * 1.1)


async def run_with_adaptation(prompts):
    controller = AdaptiveConcurrencyController(len(prompts))
    sem = asyncio.Semaphore(controller.current_concurrency)

    async def managed_call(prompt, index):
        async with sem:
            try:
                result = await call_llm(prompt, index)
                controller.completed += 1
                return result
            except Exception:
                controller.errors += 1
                raise
            finally:
                await controller.adjust()

    return await asyncio.run(*[managed_call(p, i) for i, p in enumerate(prompts)])


if __name__ == "__main__":
    import os

    file_path = (
        f"data/covid/tasks/patient_tasks_counterfactuals_covid_batch_complete.pkl"
    )

    # results = run_with_adaptation(tasks)

    results = asyncio.run(run_llm_calls(tasks))  # Run tasks properly

    with open(file_path, "wb") as f:
        pkl.dump(results, f)

    print("Collected", len(results), "responses.")

# %%
