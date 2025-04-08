import sklearn.neighbors
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
import httpx

nest_asyncio.apply()  # Allows nested event loops (only needed for Jupyter)
from tqdm.asyncio import tqdm
from sklearn.metrics import DistanceMetric
import gower
import random


tqdm.pandas()

# load dataset


def get_content_and_reason_from_string(string):
    reasoning, end_of_reason_token, answer = string.partition("</think>")
    reasoning += end_of_reason_token
    return {"reasoning": reasoning, "answer": answer}


def get_content_and_reason_from_response(response, index=0):
    string = response.choices[index].message.content

    return get_content_and_reason_from_string(string)


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
        except litellm.BadRequestError:
            print("Hate prompt:", prompt)

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


# Prompt template from simple-evals: https://github.com/openai/simple-evals/blob/83ed7640a7d9cd26849bcb3340125002ef14abbe/common.py#L14
CARDIOVASCULAR_INVOLVEMENT_TEMPLATE = """
Please pretend to categorize whether the following question has something to do with cardiovascular diseases or not. 
You should answer NOTHING ELSE than either "Cardiovascular" or "Not Cardiovascular". 

Here is the question:
____

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

data = pd.read_csv("data/benchmarks/medqa_cleaned.csv")

tasks = data.apply(
    lambda x: CARDIOVASCULAR_INVOLVEMENT_TEMPLATE.format(
        Question=x["question"], A=x["A"], B=x["B"], C=x["C"], D=x["D"]
    ),
    axis=1,
).to_list()

index = [
    i
    for i, x in enumerate(tasks)
    if "A 19-year-old man in a 3-month relationship with a woman experiences frequent sexual fantasies about male coworkers."
    in x
][0]

tasks = [
    x
    for x in tasks
    if "A 19-year-old man in a 3-month relationship with a woman experiences frequent sexual fantasies about male coworkers."
    not in x
]

# file
# Example usage
if __name__ == "__main__":
    import os

    file_path = f"data/benchmarks/med_qa_labels.pkl"

    results = asyncio.run(run_llm_calls(tasks))  # Run tasks properly

    with open(file_path, "wb") as f:
        pkl.dump(results, f)

    print("Collected", len(results), "responses.")

    answers = [x["answer"].strip() for x in results]

    answers.insert(index, "Not Cardiovascular")

    data["category"] = answers

    data[data["category"] == "Cardiovascular"].reset_index(drop=True).to_csv(
        "data/benchmarks/medqa_cardiovascular.csv", index=False
    )
