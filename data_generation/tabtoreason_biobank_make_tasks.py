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

data = pd.read_excel("../tabular_datasets/biobank_cvd.xlsx")

# remove nonsense columns

data = data[[x for x in data.columns if "AUROC" not in x]]

column_names = list(data.columns)

convert_columns_prompt = f"""You are a powerful AI with expertise in medicine. 
You are given a dataset with columns that relate to patients where each patient is a row and each column contains different information pertaining to the patient.
As your first task, you are tasking with converting a list of column names that are possibly abbreviated or not easy to understand into a fully understandable name for medical professionals.
Please provide the output as a Python dictionary. 
The list of column names is: {column_names}"""

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
distances = np.load("distances.npy")

dictionaries = X.apply(lambda x: dict(x), axis=1)

from_json_to_question_prompt = """You are a powerful AI with expertise in medicine. 
Your task is to generate a detailed and exhaustive text description for a patient.  
You are given all the patient information in a json-format, which contains the clinical attributes and the results from laboratory tests from real world patients. 
The patients in question are patients with cardiovascular disease.  
The reader of the description is an expert witin this particular medical domain. 
The language used in the description should reflect your domain expertise and your medical reasoning capabilities.
Please provide as many details as possible.
You should ONLY include the patient description!
_____
The json-file containing the information from the patient:\n"""

patient_description_prompts = [
    f"{FROM_JSON_TO_QUESTION_PROMPT}{json_representation}"
    for json_representation in dictionaries
]


all_responses_list = []

for batch_index in range(10):
    with open(
        f"data/biobank/patient_descriptions_biobank_batch_{batch_index}.pkl", "rb"
    ) as f:
        file = pkl.load(f)
        all_responses_list.extend(file)

reasoning_responses = [x["reasoning"] for x in all_responses_list]
answer_responses = [x["answer"] for x in all_responses_list]

response_data = pd.DataFrame(
    {"reasoning": reasoning_responses, "answers": answer_responses, "outcome": y}
)


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

# sns.displot(data=response_data, x="difficulty")

system_prompt_for_tasks = """You are a powerful AI with expertise in medicine with a specialty in cardiovascular patients. 
You are given three patient descriptions and their outcome (whether they died or not).
Your **target patient** is the third and final patient.
The two first patient descriptions are **example patients**. 
These patients are similar to the target patient. One of the example patients died and the other did not.
Using the two example patients, you have two tasks.
The first is to *explain* and *reason clinically* about why the target patient died or not. 
The second is to determine whether you think the label of the target patient is correct. 
The reader of your response is an expert witin this particular medical domain.
Your response should reflect your deep domain expertise and your medical reasoning capabilities.
Please provide the most well-reasoned response that you can.
Your answers should strictly be in the following format:
1) Correct label: <Yes/No>
2) Reason for label: <explanation for why the patient died or not>
_____
Example Patient 1: 
Patient Description: {example_patient_1_description} 
**Outcome: {example_patient_1_outcome}**

Example Patient 2: 
Patient Description: {example_patient_2_description} 
**Outcome: {example_patient_2_outcome}**

Target Patient:
Patient Description: {target_patient_description}
**Outcome: {target_patient_outcome}**
"""


def generate_tasks_from_patient_descriptions(row, answer_responses, system_prompt):
    target_patient_outcome, target_patient_description = (
        row["outcome_text"],
        row["answers"],
    )
    patient_description_0 = answer_responses[row["closest_sample_0"]]
    patient_outcome_0 = "Alive"
    patient_description_1 = answer_responses[row["closest_sample_1"]]
    patient_outcome_1 = "Dead"

    task_prompt = system_prompt.format(
        example_patient_1_description=patient_description_0,
        example_patient_1_outcome=patient_outcome_0,
        example_patient_2_description=patient_description_1,
        example_patient_2_outcome=patient_outcome_1,
        target_patient_description=target_patient_description,
        target_patient_outcome=target_patient_outcome,
    )
    return task_prompt


tasks = response_data.apply(
    lambda x: generate_tasks_from_patient_descriptions(
        x, answer_responses, system_prompt_for_tasks
    ),
    axis=1,
).to_list()


all_responses_list = []

for batch_index in range(10):
    with open(f"data/biobank/tasks/patient_tasks_biobank_batch_{batch_index}.pkl", "rb") as f:
        file = pkl.load(f)
        all_responses_list.extend(file)

reasoning_responses = [x["reasoning"] for x in all_responses_list]
answer_responses = [x["answer"] for x in all_responses_list]

response_data["reasoning_task"] = reasoning_responses
response_data["answers_task"] = answer_responses
response_data["task"] = tasks

response_data["reasoning_task_length"] = response_data["reasoning_task"].apply(
    lambda x: len(x)
)

sns.displot(data=response_data, x="reasoning_task_length")

sns.lmplot(data=response_data, x="difficulty", y="reasoning_task_length", scatter=True)

# file
# Example usage
if __name__ == "__main__":
    import os

    for iteration in range(10):
        print("Running batch:", iteration)
        file_path = f"data/biobank/patient_descriptions/patient_tasks_biobank_batch_{iteration}.pkl"
        if os.path.exists(file_path):
            continue
        else:
            current_tasks = tasks[iteration * 1000 : (iteration + 1) * 1000]

            results = asyncio.run(run_llm_calls(current_tasks))  # Run tasks properly

            with open(file_path, "wb") as f:
                pkl.dump(results, f)

            print("Collected", len(results), "responses.")
