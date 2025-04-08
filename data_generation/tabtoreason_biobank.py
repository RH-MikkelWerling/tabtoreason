import sklearn.neighbors
from constants import *
from openai import OpenAI, AzureOpenAI
import pandas as pd
import numpy as np
from litellm import completion, batch_completion, acompletion
#from ucimlrepo import fetch_ucirepo, list_available_datasets
from tqdm import tqdm
import asyncio
import litellm
import nest_asyncio  # Fix for Jupyter Notebooks
import pickle as pkl
import httpx

import huggingface_hub


nest_asyncio.apply()  # Allows nested event loops (only needed for Jupyter)
from tqdm.asyncio import tqdm
from sklearn.metrics import DistanceMetric
import gower
import random

tqdm.pandas()

# load dataset

data = pd.read_excel("../tabular_datasets/biobank_cvd.xlsx")

# remove nonsense columns

data = data[[x for x in data.columns if "AUROC" not in x]]

data["event"].value_counts()


def get_content_and_reason_from_string(string):
    reasoning, end_of_reason_token, answer = string.partition("</think>")
    reasoning += end_of_reason_token
    return {"reasoning": reasoning, "answer": answer}


def get_content_and_reason_from_response(response, index=0):
    string = response.choices[index].message.content

    return get_content_and_reason_from_string(string)


column_names = list(data.columns)

convert_columns_prompt = f"""You are a powerful AI with expertise in medicine. 
You are given a dataset with columns that relate to patients where each patient is a row and each column contains different information pertaining to the patient.
As your first task, you are tasking with converting a list of column names that are possibly abbreviated or not easy to understand into a fully understandable name for medical professionals.
Please provide the output as a Python dictionary. 
The list of column names is: {column_names}"""


def prompt_model(prompt):

    messages = [{"content": prompt, "role": "user"}]

    response = completion(
        model="azure_ai/DeepSeek-R1-ndpvr",
        messages=messages,
        base_url=BASE_URL,
        api_key=API_KEY,
    )

    return get_content_and_reason_from_response(response)


async def async_prompt_model(prompt):

    messages = [{"content": prompt, "role": "user"}]

    response = await acompletion(
        model="azure_ai/DeepSeek-R1-ndpvr",
        messages=messages,
        base_url=BASE_URL,
        api_key=API_KEY,
        timeout=3000,
    )

    return response


# column_mapping = prompt_model(convert_columns_prompt)

# print(column_mapping["answer"])

dictionary_to_clinical_names = {
    "Sex": "Sex",
    "Age": "Age (Years)",
    "Weight (kg)": "Weight (Kilograms)",
    "Height (cm)": "Height (Centimeters)",
    "Smoking amount (cigarettes/day)": "Smoking Amount (Cigarettes per Day)",
    "Atrial fibrillation": "Atrial Fibrillation Diagnosis",
    "Chronic kidney disease": "Chronic Kidney Disease Diagnosis",
    "Rheumatoid arthritis": "Rheumatoid Arthritis Diagnosis",
    "Drug status: Anti-diabetic": "Anti-diabetic Medication Use",
    "Drug status: Anti-hypertensives": "Anti-hypertensive Medication Use",
    "History of diabetes": "Diabetes History",
    "Drug status: Lipid-lowering": "Lipid-lowering Medication Use",
    "Drug status: Birth Control Pill": "Oral Contraceptive Use",
    "Glucose (mmol/l)": "Blood Glucose Level (mmol/L)",
    "HbA1c (%)": "Hemoglobin A1c (HbA1c) Percentage",
    "White cell count (x10^9/l)": "White Blood Cell Count (x10^9/L)",
    "Creatinine (\\B5mol/l)": "Serum Creatinine (μmol/L)",
    "Triglycerides (mmol/l)": "Triglyceride Level (mmol/L)",
    "Uric acid (\\B5mol/l)": "Uric Acid Level (μmol/L)",
    "Cystatin-c (mg/l)": "Cystatin C Level (mg/L)",
    "SBP (mmHg)": "Systolic Blood Pressure (mmHg)",
    "Urine Microalbumin (mg/L)": "Urine Microalbumin Concentration (mg/L)",
    "CRP (mg/l)": "C-Reactive Protein (CRP) Level (mg/L)",
    "Familiy history of CVD": "Family History of Cardiovascular Disease (CVD)",
    "Drug status: atypical antipsychotic medication": "Atypical Antipsychotic Medication Use",
    "Drug status: steroid tablets": "Corticosteroid Medication Use",
    "Do you have migraines?": "Migraine History",
    "Severe mental illness?": "Severe Mental Illness Diagnosis",
    "Systemic lupus erythematosus (SLE)": "Systemic Lupus Erythematosus (SLE) Diagnosis",
    "Total cholesterol": "Total Cholesterol Level (mmol/L)",
    "HDL": "High-Density Lipoprotein (HDL) Cholesterol Level (mmol/L)",
    "Ethnicity": "Ethnicity",
    "event": "Clinical Event Occurrence",
    "time_to_event": "Time to Clinical Event (Days)",
}

renamed_data = data.rename(columns=dictionary_to_clinical_names)

renamed_data = renamed_data.drop_duplicates(
    subset=[
        x
        for x in renamed_data.columns
        if x != "Clinical Event Occurrence" and x != "Time to Clinical Event (Days)"
    ]
).reset_index(drop=True)

# so here, what we need to do is to exclude the first 5000 from each class, and then let's get to 50000

renamed_data = renamed_data.reset_index()

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

already_processed_patients = trace_data["index"].values

renamed_data_filtered = renamed_data[
    ~renamed_data["index"].isin(already_processed_patients)
]

positives = renamed_data_filtered[
    renamed_data_filtered["Clinical Event Occurrence"] == 1
].reset_index(drop=True)

n_positives = len(positives)

negatives = (
    renamed_data_filtered[renamed_data_filtered["Clinical Event Occurrence"] == 0]
    .head(40000 - n_positives)
    .reset_index(drop=True)
)

trace_data = pd.concat([positives, negatives]).reset_index(drop=True)


y = trace_data["Clinical Event Occurrence"]

X = trace_data[
    [
        x
        for x in trace_data.columns
        if x != "Clinical Event Occurrence"
        and x != "Time to Clinical Event (Days)"
        and x != "index"
    ]
]

# distances_parallel = sklearn.metrics.pairwise_distances(
#     X, metric=gower.gower_dist.gower_get, n_jobs=6
# )

# sklearn.metrics.pairwise.distance_metrics()

# get distances
# distances = gower.gower_matrix(X)
# np.save("distances.npy", distances)
# distances = np.load("distances.npy")

# should save the matrix


def concatenate_description_and_question(description, question):
    return description + "\n_____________\n" + question


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
    f"{from_json_to_question_prompt}{json_representation}"
    for json_representation in dictionaries
]


# Limit concurrent requests to 150
sem = asyncio.Semaphore(500)


async def call_llm(prompt, index, progress_queue, max_retries=10):
    """Asynchronously calls the LLM with exponential backoff on rate limits."""
    messages = [{"content": prompt, "role": "user"}]
    for attempt in range(max_retries):
        try:
            response = await litellm.acompletion(
                model="azure_ai/DeepSeek-R1-ndpvr",
                messages=messages,
                base_url=BASE_URL,
                api_key=API_KEY,
                timeout=300,
                # stream=True,
            )
            # content_list = []
            # async for chunk in response:
            #     if chunk["choices"][0]["delta"]["content"]:
            #         content_list.append(chunk["choices"][0]["delta"]["content"])
            # final_string = "".join(content_list)

            await progress_queue.put(1)  # Notify progress update
            return index, get_content_and_reason_from_response(response)

        except (litellm.APIError, httpx.ConnectError) as e:
            wait_time = 2**attempt + random.uniform(0, 1)
            print(
                f"Connection issue: {e}. Retrying in {wait_time:.2f} seconds... (Attempt {attempt+1}/{max_retries})"
            )
            await asyncio.sleep(wait_time)  # Exponential backoff

        except litellm.RateLimitError:
            wait_time = 2**attempt + random.uniform(0, 1)  # Exponential backoff
            print(
                f"Rate limit hit. Retrying in {wait_time:.2f} seconds... (Attempt {attempt+1}/{max_retries})"
            )
            await asyncio.sleep(wait_time)
        except litellm.Timeout:
            wait_time = min(5 * (attempt + 1), 30)
            print(
                f"Timeout occurred. Retrying in {wait_time:.2f}s... (Attempt {attempt+1}/{max_retries})"
            )
            await asyncio.sleep(wait_time)
        except (litellm.APIError, httpx.ConnectError) as e:
            wait_time = 2**attempt + random.uniform(0, 1)
            print(
                f"Connection issue: {e}. Retrying in {wait_time:.2f} seconds... (Attempt {attempt+1}/{max_retries})"
            )
            await asyncio.sleep(wait_time)  # Exponential backoff

    print(f"Max retries reached for prompt: {prompt}")
    return None  # Return None if all retries fail


async def limited_call_llm(prompt, index, progress_queue):
    """Limits the number of concurrent requests using Semaphore and adds delays."""
    async with sem:
        await asyncio.sleep(0.5)  # Small delay to prevent overload
        return await call_llm(prompt, index, progress_queue)


async def run_llm_calls(prompts):
    """Runs multiple LLM calls in parallel with limited concurrency."""
    tasks = [limited_call_llm(prompt) for prompt in prompts]  # Defines tasks

    results = [None] * len(tasks)
    with tqdm(total=len(prompts), desc="LLM Calls Completed") as pbar:
        for i, result in enumerate(
            await asyncio.gather(*tasks)
        ):  # coro in asyncio.as_completed(tasks, timeout=3000):  # Executes them properly
            result[i] = result
            pbar.update(1)
            # result = await coro
            # results.append(result)
            # pbar.update(1)  # Update progress bar

    return results  # Returns results instead of printing


async def run_llm_calls(prompts):
    """Runs multiple LLM calls in parallel, preserving order & showing progress."""
    progress_queue = asyncio.Queue()  # Queue for tracking progress
    tasks = [
        limited_call_llm(prompt, i, progress_queue) for i, prompt in enumerate(prompts)
    ]

    results = [None] * len(prompts)  # Placeholder for ordered results

    async def track_progress(progress_queue, total):
        """Updates the progress bar as LLM calls complete."""
        with tqdm(total=total, desc="LLM Calls Completed") as pbar:
            for _ in range(total):
                await progress_queue.get()  # Wait for a task to signal completion
                pbar.update(1)  # Update progress bar

    # Run LLM calls and progress tracker in parallel
    asyncio.create_task(
        track_progress(progress_queue, len(prompts))
    )  # Run progress bar asynchronously
    results_data = await asyncio.gather(*tasks)

    # Store results in correct order
    for index, response in results_data:
        results[index] = response

    return results


# file
# Example usage
if __name__ == "__main__":
    import os
    # batching currently done hardcoded
    for iteration in range(40):
        print("Running batch:", iteration)
        file_path = (
            f"data/biobank/patient_descriptions_biobank_batch_{iteration+10}.pkl"
        )
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
