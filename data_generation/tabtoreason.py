from constants import *
from openai import OpenAI, AzureOpenAI
import pandas as pd
import numpy as np
from litellm import completion, batch_completion, acompletion
from ucimlrepo import fetch_ucirepo, list_available_datasets
from tqdm import tqdm
import asyncio
import litellm
import nest_asyncio  # Fix for Jupyter Notebooks
import pickle as pkl

nest_asyncio.apply()  # Allows nested event loops (only needed for Jupyter)
from tqdm.asyncio import tqdm

# from tqdm.auto import tqdm  # for notebooks

# Create new `pandas` methods which use `tqdm` progress
# (can use tqdm_gui, optional kwargs, etc.)
tqdm.pandas()

# load dataset

support_data = pd.read_csv("../tabular_datasets/support_data.csv")

support_matrix = pd.read_csv("../tabular_datasets/distance_matrix_subset_support.csv")

# NOTE: Should distances be **without** the labels?

support_data = support_data[
    [x for x in support_data.columns if "Unnamed" not in x and "d.time" not in x]
]

support_data = support_data.head(1000)


def get_content_and_reason_from_response(response, index=0):
    reasoning, end_of_reason_token, answer = response.choices[
        index
    ].message.content.partition("</think>")
    reasoning += end_of_reason_token
    return {"reasoning": reasoning, "answer": answer}


column_names = list(support_data.columns)

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

# print(column_mapping)

dictionary_to_clinical_names = column_name_mapping = {
    "sex": "Biological Sex",
    "ARF/MOSF w/Sepsis": "Acute Renal Failure with Multiple Organ System Failure and Sepsis",
    "COPD": "Chronic Obstructive Pulmonary Disease",
    "CHF": "Congestive Heart Failure",
    "Cirrhosis": "Cirrhosis",
    "Coma": "Coma",
    "Colon Cancer": "Colon Cancer",
    "Lung Cancer": "Lung Cancer",
    "MOSF w/Malig": "Multiple Organ System Failure with Malignancy",
    "ARF/MOSF": "Acute Renal Failure with Multiple Organ System Failure",
    "Cancer": "Cancer",
    "under $11k": "Income Under 11,000 USD",
    "$11-$25k": "Income Between 11,000 and 25,000 USD",
    "$25-$50k": "Income Between 25,000 and 50,000 USD",
    ">$50k": "Income Over 50,000 USD",
    "white": "Race: White",
    "black": "Race: Black",
    "asian": "Race: Asian",
    "hispanic": "Ethnicity: Hispanic",
    "num.co": "Number of Comorbidities",
    "edu": "Education Level (Years)",
    "avtisst": "Average TISS (Therapeutic Intervention Scoring System) Score",
    "hday": "Hospital Day of Measurement",
    "diabetes": "Diabetes",
    "dementia": "Dementia",
    "meanbp": "Mean Blood Pressure",
    "wblc": "White Blood Cell Count",
    "hrt": "Heart Rate",
    "resp": "Respiratory Rate",
    "temp": "Body Temperature",
    "pafi": "PaO₂/FiO₂ Ratio",
    "alb": "Serum Albumin",
    "bili": "Bilirubin",
    "crea": "Serum Creatinine",
    "sod": "Serum Sodium",
    "ph": "Arterial pH",
    "glucose": "Blood Glucose",
    "bun": "Blood Urea Nitrogen",
    "urine": "Urine Output",
    "adlp": "ADL (Activities of Daily Living) Performance",
    "adls": "ADL (Activities of Daily Living) Score",
    "death": "Mortality Indicator",
    "d.time": "Time to Death or Day of Death",
    "age": "Age",
}

renamed_support_data = support_data.rename(columns=dictionary_to_clinical_names)

y = pd.cut(
    renamed_support_data["Age"], bins=[-np.inf, 65, np.inf], labels=["-65", "65+"]
)

X = renamed_support_data[
    [
        x
        for x in renamed_support_data.columns
        if x != "Mortality Indicator" and x != "Age"
    ]
]

y = renamed_support_data["Mortality Indicator"]

X = renamed_support_data[
    [x for x in renamed_support_data.columns if x != "Mortality Indicator"]
]


def concatenate_description_and_question(description, question):
    return description + "\n_____________\n" + question


dictionaries = X.apply(lambda x: dict(x), axis=1)

from_json_to_question_prompt = """You are a powerful AI with expertise in medicine. 
Your task is to generate a detailed text description of a patient. The reader of the description is an expert in the domain and the language used in the description should reflect your domain expertise and your medical reasoning capabilities. 
The information provided is based on clinical values from real world patients. 
The patients in question are seriously ill hospitalized adults.  
You are given all the patient information in a json-format. 
Please provide as many details as possible.
You should ONLY include the patient description!
_____
The json-file containing the information from the patient:\n"""

patient_description_prompts = [
    f"{from_json_to_question_prompt}{json_representation}"
    for json_representation in dictionaries
]

import asyncio
import litellm
import random
from tqdm.asyncio import tqdm

MAX_CONCURRENCY = 75  # Start high and adjust dynamically
MIN_CONCURRENCY = 1  # Minimum concurrency level


class AdaptiveLimiter:
    """Dynamically adjusts concurrency based on rate limit errors."""

    def __init__(self, start_concurrency=MAX_CONCURRENCY):
        self.concurrency = start_concurrency
        self.lock = asyncio.Lock()

    async def get_permit(self):
        """Waits for a permit based on current concurrency limit."""
        async with self.lock:
            if self.concurrency < 1:
                self.concurrency = 1  # Never go below 1
            return asyncio.Semaphore(self.concurrency)

    async def adjust(self, success=True):
        """Increases or decreases concurrency based on API response."""
        async with self.lock:
            if success:
                self.concurrency = min(self.concurrency + 1, MAX_CONCURRENCY)
            else:
                self.concurrency = max(
                    self.concurrency // 2, MIN_CONCURRENCY
                )  # Reduce by half if rate-limited


adaptive_limiter = AdaptiveLimiter()


async def call_llm(prompt, max_retries=10, timeout=300):
    """Calls the LLM with retries & rate limit adaptation."""
    for attempt in range(max_retries):
        try:
            response = await litellm.acompletion(
                model="azure_ai/DeepSeek-R1-ndpvr",
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout,
                base_url=BASE_URL,
                api_key=API_KEY,
            )
            await adaptive_limiter.adjust(
                success=True
            )  # Increase concurrency if successful
            return response  # Success

        except litellm.RateLimitError:
            await adaptive_limiter.adjust(
                success=False
            )  # Reduce concurrency if rate-limited
            wait_time = 2**attempt + random.uniform(0, 1)
            print(
                f"Rate limit hit. Retrying in {wait_time:.2f}s... (Attempt {attempt+1}/{max_retries})"
            )
            await asyncio.sleep(wait_time)

        except litellm.Timeout:
            wait_time = min(5 * (attempt + 1), 30)
            print(
                f"Timeout occurred. Retrying in {wait_time:.2f}s... (Attempt {attempt+1}/{max_retries})"
            )
            await asyncio.sleep(wait_time)

    return None  # Return None if all retries fail


async def limited_call_llm(prompt):
    """Controls request flow dynamically."""
    sem = await adaptive_limiter.get_permit()
    async with sem:
        return await call_llm(prompt)


async def run_llm_calls(prompts):
    """Runs multiple LLM calls in parallel with dynamic concurrency."""
    tasks = [limited_call_llm(prompt) for prompt in prompts]

    results = []
    with tqdm(total=len(prompts), desc="LLM Calls Completed") as pbar:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            pbar.update(1)

    return results  # Returns results instead of printing


# Example usage
if __name__ == "__main__":
    results = asyncio.run(
        run_llm_calls(patient_description_prompts)
    )  # Run tasks properly

    with open("patient_description_subset.pkl", "wb") as file:
        pkl.dump(results, file)

    print(f"✅ Collected {len(results)} responses.")


# Limit concurrent requests to 150
sem = asyncio.Semaphore(2)


async def call_llm(prompt, max_retries=10):
    """Asynchronously calls the LLM with exponential backoff on rate limits."""
    messages = [{"content": prompt, "role": "user"}]
    for attempt in range(max_retries):
        try:
            response = await litellm.acompletion(
                model="azure_ai/DeepSeek-R1-ndpvr",
                messages=messages,
                base_url=BASE_URL,
                api_key=API_KEY,
            )
            return response  # Successful response

        except litellm.RateLimitError:
            wait_time = 2**attempt + random.uniform(0, 1)  # Exponential backoff
            print(
                f"Rate limit hit. Retrying in {wait_time:.2f} seconds... (Attempt {attempt+1}/{max_retries})"
            )
            await asyncio.sleep(wait_time)

    print(f"Max retries reached for prompt: {prompt}")
    return None  # Return None if all retries fail


async def limited_call_llm(prompt):
    """Limits the number of concurrent requests using Semaphore."""
    async with sem:
        return await call_llm(prompt)


async def run_llm_calls(prompts):
    """Runs multiple LLM calls in parallel with limited concurrency."""
    tasks = [limited_call_llm(prompt) for prompt in prompts]  # Defines tasks

    results = []
    with tqdm(total=len(prompts), desc="LLM Calls Completed") as pbar:
        for coro in asyncio.as_completed(tasks, timeout=3000):  # Executes them properly
            result = await coro
            results.append(result)
            pbar.update(1)  # Update progress bar

    return results  # Returns results instead of printing


# Example usage
if __name__ == "__main__":
    results = asyncio.run(
        run_llm_calls(patient_description_prompts)
    )  # Run tasks properly

    print("Collected", len(results), "responses.")


def generate_questions(row, problem_description):
    json_representation = [dict(row)]

    from_json_to_question_prompt = f"""You are a powerful AI with expertise in medicine. 
    Your task is to generate a detailed text description of a patient. The reader of the description is an expert in the domain and the language used in the description should reflect your domain expertise and your medical reasoning capabilities. 
    The information provided is based on clinical values from real world patients. 
    The patients in question are seriously ill hospitalized adults.  
    You are given all the patient information in a json-format. 
    Please provide as many details as possible.
    You should ONLY include the patient description!
    _____
    The json-file containing the information from the patient: {json_representation}"""

    # problem_description = """What is the best guess for whether this patient is 65+ or -65 years old? Answers are ONLY VALID if they (a) answer in a binary fashion (65+ = 65 years or older, -65 = under 65 years old) and (b) the final answer is supplied as the last part of the response as:  \nFinal Answer: 65+/-65"""

    generate_patient_description = prompt_model(prompt=from_json_to_question_prompt)

    question = concatenate_description_and_question(
        generate_patient_description["answer"], problem_description
    )

    return question


if __name__ == "__main__":

    X["patient_question"] = X.progress_apply(
        lambda row: generate_questions(row), axis=1
    )

    X.to_csv("datasets/support_data_questions.csv")

# X.apply(print, axis = 1)

# json_representation = X.iloc[[0]].to_dict("records")
# # from_json_to_question_prompt = f"""You are a powerful AI with expertise in medicine.
# # Your task is to generate challenging questions that require medical domain reasoning based on clinical values from real world patients.
# # The questions are generated by a detailed patient description followed by a multiple choice question. Give four plausible answers (A, B, C, D) after posing the question.
# # You are given all the patient information in a json-format. First, describe the patient based on the values contained in the json in appropriate medical language.
# # The question should be what the best guess for the patient's age is given the clinical information provided. You should ONLY include the description, the question and the multiple choice answers and no other text!
# # The json-file containing the information from the patient: {json_representation}"""


# from_json_to_question_prompt = f"""You are a powerful AI with expertise in medicine.
# Your task is to generate a detailed text description of a patient. The reader of the description is an expert in the domain and the language used in the description should reflect your domain expertise and your medical reasoning capabilities.
# The information provided is based on clinical values from real world patients.
# The patients in question are seriously ill hospitalized adults.
# You are given all the patient information in a json-format.
# Please provide as many details as possible.
# You should ONLY include the patient description!
# _____
# The json-file containing the information from the patient: {json_representation}"""

# problem_description = '''What is the best guess for whether this patient is 65+ or -65 years old? Answers are ONLY VALID if they (a) answer in a binary fashion (65+ = 65 years or older, -65 = under 65 years old) and (b) the final answer is supplied as the last part of the response as:  \nFinal Answer: 65+/-65'''

# generate_patient_description = prompt_model(prompt=from_json_to_question_prompt)

# print(generate_patient_description["answer"])

# def concatenate_description_and_question(description, question):
#     return description + "\n_____________\n" + question

# question = concatenate_description_and_question(generate_patient_description["answer"], problem_description)

# answer = prompt_model(prompt=question)

# answer["answer"]

# def make_continous_discrete(variable):
#     bins = [15, 30, 40, 50, 60, np.inf]
#     labels = [f'{i}+' if j==np.inf else f'{i}-{j}' for i, j in zip(bins, bins[1:])]

#     dataset['AgeRange'] = pd.cut(dataset['Age'], bins, labels)
#     variable <
