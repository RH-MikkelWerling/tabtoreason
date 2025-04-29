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


### CONSIDER THIS FOR SIMPLE PROMPT:

system_prompt = """**Role**: Cardiovascular Specialist

**Response Format**:
[Brief medical reasoning]
Final answer: X

**Rules**:
- X must be A/B/C/D
- "Final answer:" must be the last line
- No other text/comments after"""

### CONSIDER THIS FOR SIMPLE PROMPT:

system_prompt = """**Role**: Cardiovascular Specialist

**Response Format**:
[Brief medical reasoning]
Final answer: X

**Rules**:
- X must be A/B/C/D
- "Final answer:" must be the last line
- No other text/comments after"""

system_prompt = """**Role**: Cardiovascular Specialist

**Response Format**:
[Brief medical reasoning]
Final answer: X

**Rules**:
- X must be A/B/C/D
- "Final answer:" must be the last line
- No other text/comments after"""


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

Final Answer: X

**Critical Constraints**:
- X must be A, B, C, or D
- No text/comments after final answer
- Never use markdown formatting

**Example**:
1. First analysis: 62yo male with crushing substernal chest pain radiating to jaw
2. Second analysis: ESC 2023 guidelines prioritize ECG within 10 minutes for ACS
3. Final determination: Option D delays critical diagnostics

Final Answer: B
"""


system_prompt = """**Role**: Dr. CardioAI - Cardiovascular Disease Specialist

**Task**: Analyze and answer medical questions with:
1. Concise clinical reasoning 
2. Final answer in strict format

**Response Format**:
<Your medical reasoning process>
Final answer: X

**Critical Requirements**:
- X must be A, B, C, or D
- "Final answer: " must be the last line
- No additional text/comments after final answer
- Never use markdown formatting

**Example**:
Question: What's first-line therapy for stable angina?
Options: A) Aspirin B) Nitroglycerin C) Warfarin D) Metoprolol

Response:
Beta-blockers reduce myocardial oxygen demand. Per ACC/AHA guidelines, first-line therapy for stable angina includes beta-blockers or calcium channel blockers.
Final answer: D
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


def get_stop_token_ids(tokenizer):
    stop_ids = []

    # EOS token
    if tokenizer.eos_token_id is not None:
        stop_ids.append(tokenizer.eos_token_id)

    # Chinese token (remove if using Llama)
    # stop_ids.append(tokenizer.convert_tokens_to_ids("答案"))  # Remove this line

    # Handle special tokens safely
    special_tokens = ["</s>", "\nUser:"]
    for token in special_tokens:
        try:
            ids = tokenizer.encode(token, add_special_tokens=False)
            if ids:
                stop_ids.append(ids[0])
        except:
            continue

    # Deduplicate while preserving order
    seen = set()
    return [x for x in stop_ids if x not in seen and not seen.add(x)]


def get_stop_token_ids(tokenizer):
    """Safe stop token extraction for multiple architectures"""
    stop_ids = []

    # Universal base tokens
    base_tokens = [
        tokenizer.eos_token,
        tokenizer.pad_token,
        "</s>",
        "\nUser:",
        "答案",  # For Chinese models like Qwen
        "<|endoftext|>",
        "<|im_end|>",
    ]

    # Model-agnostic collection
    for token in base_tokens:
        try:
            if token and token in tokenizer.get_vocab():
                stop_ids.append(tokenizer.convert_tokens_to_ids(token))
        except:
            continue

    # Add encoded sequences safely
    sequences = ["\nFinal answer:", "\nAnswer:", "</think>"]
    for seq in sequences:
        try:
            seq_id = tokenizer.encode(seq, add_special_tokens=False)
            if seq_id:
                stop_ids.append(seq_id[0])
        except:
            continue

    # Deduplicate and validate
    valid_ids = [int(i) for i in stop_ids if i is not None]
    return list(dict.fromkeys(valid_ids))  # Preserve order, remove duplicates


def get_stop_token_ids(tokenizer):
    """Architecture-agnostic stop token ID extraction, compatible with Qwen, LLaMA, DeepSeek."""
    candidate_tokens = [
        tokenizer.eos_token,  # Should work if eos_token is defined
        tokenizer.pad_token,  # Optional fallback
        "<|endoftext|>",  # OpenAI-style
        "<|im_end|>",  # DeepSeek / ChatML-style
        "<|EOT|>",  # Qwen-style
        "</s>",  # HuggingFace default
        "答案",  # Qwen-specific 'Answer'
        "\nUser:",  # Avoid multi-turn continuation
        "\nFinal answer:",  # CoT-style stopping
        "\nAnswer:",
        "</think>",
    ]

    stop_ids = set()

    # Try to encode each token robustly
    for token in candidate_tokens:
        if not token:
            continue
        try:
            # Try to encode full token/phrase and add ALL token IDs
            encoded = tokenizer.encode(token, add_special_tokens=False)
            stop_ids.update(encoded)
        except Exception:
            continue

    # Also include explicit eos_token_id if not caught above
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)

    return list(stop_ids)


def get_stop_token_ids(tokenizer):
    """
    Safe, architecture-agnostic stop token collection for QA evaluation.
    Only includes true stop markers (not semantic tokens like 'Answer:').
    """
    candidate_tokens = [
        tokenizer.eos_token,
        tokenizer.pad_token,
        "<|endoftext|>",
        "<|im_end|>",
        "<|EOT|>",
        "</s>",
    ]

    stop_ids = set()

    for token in candidate_tokens:
        if not token:
            continue
        try:
            encoded = tokenizer.encode(token, add_special_tokens=False)
            stop_ids.update(encoded)
        except Exception:
            continue

    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)

    return list(stop_ids)


def get_stop_token_ids(tokenizer):
    return [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|EOT|>"),  # Common in chat models
        tokenizer.convert_tokens_to_ids("</s>"),
        tokenizer.encode("\nUser:")[0],  # Prevent multi-turn hallucinations
    ]


def get_model_family(tokenizer):
    """
    Heuristically determine the model family based on tokenizer attributes.
    Returns one of: "qwen", "llama", or "default"
    """
    name = tokenizer.name_or_path.lower()

    if "qwen" in name or "<|eot|>" in tokenizer.get_vocab():
        return "qwen"
    elif "llama" in name or "llama" in tokenizer.__class__.__name__.lower():
        return "llama"
    else:
        return "default"


def get_stop_token_ids(tokenizer):
    """
    Return stop token IDs based on the model family (Qwen, LLaMA, etc.)
    """
    model_family = get_model_family(tokenizer)
    stop_ids = set()

    # 1. Always include eos_token_id if available
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)

    # 2. Model-specific additions
    if model_family == "qwen":
        tokens = ["<|EOT|>", "</s>"]
    elif model_family == "llama":
        tokens = ["</s>", "<|endoftext|>"]
    else:  # default fallback
        tokens = ["</s>", "<|endoftext|>", "<|im_end|>"]

    for tok in tokens:
        try:
            tok_id = tokenizer.convert_tokens_to_ids(tok)
            if tok_id is not None and tok_id != tokenizer.unk_token_id:
                stop_ids.add(tok_id)
        except Exception:
            continue

    # 3. Prevent multi-turn hallucinations (optional heuristic)
    try:
        user_tok = tokenizer.encode("\nUser:", add_special_tokens=False)
        if user_tok:
            stop_ids.add(user_tok[0])
    except:
        pass

    return list(stop_ids)


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
    max_model_len: int = 32768,
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
    match = re.search(
        r"^Final answer:\s*([A-D])$", response, re.MULTILINE | re.IGNORECASE
    )
    return match.group(1).upper() if match else None


def generate_completions(
    model: LLM,
    tokenizer,
    prompts: list[str],
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 32768,
    n: int = 1,
):
    """Generate completions with proper output unpacking"""
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=n,
        frequency_penalty=1.2,
        presence_penalty=0.9,
        stop_token_ids=get_stop_token_ids(tokenizer),
        min_p=0.05,
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
    return answers, reasonings, answer_texts, all_completions, failed_indices


dataframe = pd.DataFrame(dataset_test["train"])

# Example usage
if __name__ == "__main__":
    # Initialize model (replace with your HF model ID)

    models = [
        # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        # "mikkel-werling/DeepSeek-R1-Distill-Qwen-1.5B-Patient-Descriptions",
        "mikkel-werling/DeepSeek-R1-Distill-Qwen-1.5B-Tasks",
        # "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        # "mikkel-werling/DeepSeek-R1-Distill-Qwen-7B-Patient-Descriptions",
        # "mikkel-werling/DeepSeek-R1-Distill-Qwen-7B-Tasks",
        # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    ]

    for model_path in models:

        specific_prompts = [
            format_with_chat_template(prompt, model_path) for prompt in prompts
        ]

        print(f"NOW RUNNING: {model_path}")
        model_name = model_path.split("/")[-1]
        model = load_vllm_model(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Generate answers

        answers, reasonings, answer_texts, completions, failed = generate_until_valid(
            model=model,
            tokenizer=tokenizer,
            prompts=specific_prompts,
            temperature=0.6,
        )

        # Access parsed components
        for idx in range(len(answers)):
            print(f"Question {idx}:")
            print("Reasoning:\n", reasonings[idx])
            print("Answer Section:\n", answer_texts[idx])
            print("Extracted Answer:", answers[idx])
            print("\n")

        dataframe["model_answer"] = answer_texts
        dataframe["model_reasoning"] = reasonings
        dataframe["full_responses"] = completions
        dataframe["extracted_answers"] = answers

        dataframe.to_csv(
            f"data/biobank/processed_data/{model_name}_responses.csv", index=False
        )
