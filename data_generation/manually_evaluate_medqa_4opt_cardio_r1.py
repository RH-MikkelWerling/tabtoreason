from constants import *
from prompt_functions import *
import pandas as pd
import numpy as np
from tqdm import tqdm
import asyncio
import nest_asyncio  # Fix for Jupyter Notebooks
import pickle as pkl
import gc

from vllm.distributed.parallel_state import destroy_model_parallel

nest_asyncio.apply()  # Allows nested event loops (only needed for Jupyter)
from tqdm.asyncio import tqdm
import gower
import seaborn as sns

from datasets import load_dataset, DatasetDict, Dataset
import pickle as pkl

from vllm import LLM, SamplingParams
import time
import torch


tqdm.pandas()

dataset_test = load_dataset("mikkel-werling/medqa_cardiovascular")

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

MEDICAL_QA_PROMPT = """**Question**: {question}

**Options**:
A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}"""

prompts = [
    MEDICAL_QA_PROMPT.format(
        question=x["question"],
        option_a=x["A"],
        option_b=x["B"],
        option_c=x["C"],
        option_d=x["D"],
    )
    for x in dataset_test["train"]
]

# MODEL_ARGS = "pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_num_batched_tokens=32768,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

from transformers import AutoTokenizer


def get_stop_token_ids(tokenizer):
    return [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|EOT|>"),  # Common in chat models
        tokenizer.convert_tokens_to_ids("</s>"),
        tokenizer.encode("\nUser:")[0],  # Prevent multi-turn hallucinations
    ]


def anti_repetition_params(tokenizer):
    return SamplingParams(
        temperature=0.7,
        top_p=0.9,
        frequency_penalty=1.2,  # Strong repetition suppression
        presence_penalty=0.9,
        stop_token_ids=get_stop_token_ids(tokenizer),
        max_tokens=1024,
        min_p=0.05,  # New in vLLM 0.4.2 - helps prevent degenerate outputs
    )


def format_with_chat_template(question: str) -> str:
    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create the message structure your model expects
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },  # Include if used in fine-tuning
        {"role": "user", "content": question},
        {"role": "assistant", "content": ""},  # Leave empty for generation
    ]

    return messages


if __name__ == "__main__":

    responses = asyncio.run(run_llm_calls(prompts))  # Run tasks properly

    responses = [get_content_and_reason_from_string(x) for x in responses]

    answers = [x["answer"] for x in responses]

    reasoning = [x["reasoning"] for x in responses]

    dataframe["model_answer"] = answers
    dataframe["model_reasoning"] = reasoning
    dataframe["full_responses"] = responses

    dataframe["outputs"] = outputs

    dataframe.to_csv(
        f"data/biobank/processed_data/{model_name}_responses.csv", index=False
    )

    # Display results
    for question, answer in zip(prompts, responses):
        print(f"Question: {question}\nAnswer: {answer}\n{'-'*50}")
