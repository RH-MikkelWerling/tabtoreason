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
import re


tqdm.pandas()

dataset_test = load_dataset("mikkel-werling/medqa_cardiovascular")

system_prompt = """**Role**: Dr. CardioAI – Cardiovascular Disease Specialist

**Task**: Analyze and answer medical questions by:
1. Thinking through the clinical scenario in <think>...</think> tags
2. Providing the final answer after </think>

**Response Format**:
<think>
[Your step-by-step reasoning here]
</think>
Answer: X

**Critical Requirements**:
- X must be one of: A, B, C, or D
- The answer **must be the last line** of your response.
- Once you output the final answer, **stop immediately** and do not include any additional comments, reasoning, or explanations.

**Example**:
**Question**: What's first-line therapy for stable angina?

**Options**: 
A) Aspirin 
B) Nitroglycerin 
C) Warfarin 
D) Metoprolol

Response:
Answer: D
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

from transformers import AutoTokenizer


def get_stop_token_ids(tokenizer):
    """Minimal stop token set to avoid truncating valid final answers."""
    stop_ids = set()

    # Always include EOS if defined
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)

    # Include <|EOT|> *only if* it's registered in the vocab
    try:
        tok_id = tokenizer.convert_tokens_to_ids("<|EOT|>")
        if tok_id != tokenizer.unk_token_id:
            stop_ids.add(tok_id)
    except:
        pass

    # Do NOT include "</s>", "<|endoftext|>", "\nUser:", or "<|im_end|>" here
    # These may occur *after* or *within* valid answers

    return list(stop_ids)


def format_with_chat_template(question: str, model_name: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create the message structure your model expects
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },  # Include if used in fine-tuning
        {"role": "user", "content": question},
        {"role": "assistant", "content": ""},  # Leave empty for generation
    ]

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# 5. Post-process using training-aware cleaning
def training_aware_cleaner(text: str) -> str:
    # Remove any template leakage from training
    return text.split(tokenizer.eos_token)[0].replace("Assistant: ", "").strip()


def load_vllm_model(
    model_name: str,
    tensor_parallel_size: int = 4,
    dtype: str = "bfloat16",
    max_model_len: int = 8192,
) -> LLM:
    """Initialize the vLLM model"""
    return LLM(
        model=model_name,
        tokenizer=model_name,  # Only needed if different from model
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        max_model_len=max_model_len,
        trust_remote_code=True,  # Needed for custom models
    )


def extract_answer(response):
    # Match last line starting with "Final answer: "
    match = re.search(r"^Answer:\s*([A-D])$", response, re.MULTILINE | re.IGNORECASE)
    return match.group(1).upper() if match else None


def extract_answer(response):
    # Find ALL answers, take last valid one
    matches = re.findall(r"Answer:\s*([A-D])", response, re.IGNORECASE)
    for ans in reversed(matches):
        if ans.upper() in {"A", "B", "C", "D"}:
            return ans.upper()
    return None


def generate_completions(
    model: LLM,
    tokenizer,
    prompts: list[str],
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 8192,
    n: int = 1,
):
    """Generate completions with proper output unpacking"""
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=n,
        # frequency_penalty=1.2,
        # presence_penalty=0.9,
        stop_token_ids=get_stop_token_ids(tokenizer),
        # min_p=0.05,
    )

    start_time = time.time()
    request_outputs = model.generate(prompts, sampling_params)
    elapsed = time.time() - start_time

    # Extract text from each RequestOutput
    completions = []
    for req_output in request_outputs:
        # Get all generated texts for this request
        req_completions = [output.text for output in req_output.outputs]
        completions.append(req_completions)

    print(f"Generated {len(prompts)} prompts in {elapsed:.2f}s")
    return completions, request_outputs


def get_content_and_reason_from_string(string: str) -> dict:
    """Your existing extraction function"""
    reasoning, end_of_reason_token, answer = string.partition("</think>")
    reasoning += end_of_reason_token  # Preserve the closing tag if needed
    return {"reasoning": reasoning.strip(), "answer": answer.strip()}


def generate_until_valid(
    model: LLM,
    tokenizer,
    prompts: list[str],
    **generation_kwargs,
):
    """
    Modified to use your <think> tag parsing
    Returns:
        - answers: List of extracted letter answers
        - reasonings: List of full <think> blocks
        - answer_texts: List of text after </think>
        - all_completions: All generated texts
        - failed_indices: Indices of failed prompts
    """
    answers = [None] * len(prompts)
    reasonings = [None] * len(prompts)
    answer_texts = [None] * len(prompts)
    all_completions = [[] for _ in range(len(prompts))]
    remaining_indices = list(range(len(prompts)))
    attempt = 0

    while remaining_indices:

        current_prompts = [prompts[i] for i in remaining_indices]

        # Generate responses
        batch_completions, batch_outputs = generate_completions(
            model, tokenizer, current_prompts, **generation_kwargs
        )

        new_remaining = []
        for idx, (comps, out) in enumerate(zip(batch_completions, batch_outputs)):
            original_idx = remaining_indices[idx]
            all_completions[original_idx] = comps

            # Process each completion
            for comp in comps:
                parsed = get_content_and_reason_from_string(comp)
                answer = extract_answer(
                    parsed["answer"]
                )  # Use your existing answer extraction

                if answer and answer in {"A", "B", "C", "D"}:
                    # Store all components
                    answers[original_idx] = answer
                    reasonings[original_idx] = parsed["reasoning"]
                    answer_texts[original_idx] = parsed["answer"]
                    break

            if answers[original_idx] is None:
                new_remaining.append(original_idx)

        remaining_indices = new_remaining
        print(f"Retry {attempt+1}: {len(remaining_indices)} failures remaining")
        attempt += 1

    failed_indices = [i for i, ans in enumerate(answers) if ans is None]

    # After each model evaluation:
    del model
    torch.cuda.empty_cache()
    destroy_model_parallel()  # vLLM-specific cleanup
    gc.collect()
    return answers, reasonings, answer_texts, all_completions, failed_indices


dataframe = pd.DataFrame(dataset_test["train"])

# Example usage
if __name__ == "__main__":
    # Initialize model (replace with your HF model ID)

    models = [
        # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        # "mikkel-werling/DeepSeek-R1-Distill-Qwen-1.5B-Patient-Descriptions",
        # "mikkel-werling/DeepSeek-R1-Distill-Qwen-1.5B-Tasks",
        # "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        # "mikkel-werling/DeepSeek-R1-Distill-Qwen-7B-Patient-Descriptions",
        # "mikkel-werling/DeepSeek-R1-Distill-Qwen-7B-Tasks",
        # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        # "mikkel-werling/DeepSeek-R1-Distill-Llama-8B-Patient-Descriptions",
        # "mikkel-werling/DeepSeek-R1-Distill-Llama-8B-Tasks",
        "HPAI-BSC/Llama3-Aloe-8B-Alpha",
    ]

    for model_path in models:

        list_of_dataframes = []

        specific_prompts = [
            format_with_chat_template(prompt, model_path) for prompt in prompts
        ]

        print(f"NOW RUNNING: {model_path}")
        model_name = model_path.split("/")[-1]
        model = load_vllm_model(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Generate answers

        for run in range(10):

            current_dataframe = dataframe.copy()

            answers, reasonings, answer_texts, completions, failed = (
                generate_until_valid(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=specific_prompts,
                    temperature=0.8,
                )
            )

            # Access parsed components
            for idx in range(len(answers)):
                print(f"Question {idx}:")
                print("Full completion:", completions[idx])
                print("Reasoning:\n", reasonings[idx])
                print("Answer Section:\n", answer_texts[idx])
                print("Extracted Answer:", answers[idx])
                print("\n")

            current_dataframe["model_answer"] = answer_texts
            current_dataframe["model_reasoning"] = reasonings
            current_dataframe["full_responses"] = completions
            current_dataframe["extracted_answers"] = answers

            current_dataframe["run"] = run

            list_of_dataframes.append(current_dataframe)

        all_dataframes = pd.concat(list_of_dataframes).reset_index(drop=True)

        all_dataframes.to_csv(
            f"data/biobank/processed_data/{model_name}_responses_new_settings.csv",
            index=False,
        )
