from constants import *
import pandas as pd
from litellm import completion, acompletion

# from ucimlrepo import fetch_ucirepo, list_available_datasets
from tqdm import tqdm
import asyncio
import litellm
import nest_asyncio  # Fix for Jupyter Notebooks
import pickle as pkl
import httpx
from tqdm.asyncio import tqdm
import gower
import random
import os


def get_content_and_reason_from_string(string):
    reasoning, end_of_reason_token, answer = string.partition("</think>")
    reasoning += end_of_reason_token
    return {"reasoning": reasoning, "answer": answer}


def get_content_and_reason_from_response(response, index=0):
    string = response.choices[index].message.content

    return get_content_and_reason_from_string(string)


def concatenate_description_and_question(description, question):
    return description + "\n_____________\n" + question


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


# Limit concurrent requests to 500
# EXPERIMENTALLY CHANGED TO 5000
sem = asyncio.Semaphore(1000)


async def call_llm(prompt, index, progress_queue, max_retries=15):
    """Asynchronously calls the LLM with exponential backoff on rate limits."""
    system_prompt = """**Role**: You are Dr. CardioAI - a Cardiovascular Disease Specialist

    **Task Protocol**:
    1. Perform 3-step clinical analysis
    2. Select the single best answer (A-D)
    3. Follow exact output format

    **Analysis Framework**:
    1. First: Identify key clinical features
    2. Second: Apply guidelines/pathophysiology
    3. Third: Eliminate incorrect options

    **Required Format**:
    1. First analysis: [Key clinical/pathophysiological factor]
    2. Second analysis: [Guideline/evidence application] 
    3. Final determination: [Option elimination rationale]

    Answer: X

    **Critical Constraints**:
    - X must be A, B, C, or D
    - No text/comments after final answer
    - Never use markdown formatting

    **Example**:
    1. First analysis: 62yo male with crushing substernal chest pain radiating to jaw
    2. Second analysis: ESC 2023 guidelines prioritize ECG within 10 minutes for ACS
    3. Final determination: Option D delays critical diagnostics

    Answer: B
    """

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },  # Include if used in fine-tuning
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": ""},  # Leave empty for generation
    ]
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

            response_dict = get_content_and_reason_from_response(response)
            if response_dict["answer"]:

                await progress_queue.put(1)  # Notify progress update

                return index, response_dict
            else:
                wait_time = min(5 * (attempt + 1), 30)
                print(
                    f"Answer was not produced. Retrying in {wait_time:.2f} seconds... (Attempt {attempt+1}/{max_retries})"
                )
                continue

        except (litellm.APIError, httpx.ConnectError) as e:
            wait_time = 2**attempt + random.uniform(0, 1)
            print(
                f"Connection issue: {e}. Retrying in {wait_time:.2f} seconds... (Attempt {attempt+1}/{max_retries})"
            )
            await asyncio.sleep(wait_time)  # Exponential backoff

        except litellm.RateLimitError:
            wait_time = min(5 * (attempt + 1), 30)  # Exponential backoff
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
        except KeyError as e:
            wait_time = 2**attempt + random.uniform(0, 1)
            print(
                f"KeyError issue: {e} for {index}. Retrying in {wait_time:.2f} seconds... (Attempt {attempt+1}/{max_retries})"
            )
            await asyncio.sleep(wait_time)  # Exponential backoff
        except litellm.exceptions.APIConnectionError as e:
            wait_time = 2**attempt + random.uniform(0, 1)
            print(
                f"API connection issue: {e} for index {index}. Retrying in {wait_time:.2f} seconds... (Attempt {attempt+1}/{max_retries})"
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
