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

MEDICAL_QA_PROMPT = """{question}

**Options**:
A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}

**Required Response Format**:
[REASONING]
1. First reasoning step...
2. Second reasoning step...
3. Final synthesis...
[/REASONING]

Answer: <LETTER>"""

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


def format_with_chat_template(question: str, model_name: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create the message structure your model expects
    messages = [
        {
            "role": "system",
            "content": """You are Dr. CardioAI, an expert cardiologist and medical researcher. 
Your task is to analyze complex cardiovascular cases with rigorous clinical reasoning and provide evidence-based answers.

**Instructions**:
1. First, perform a detailed step-by-step analysis of the question
2. Consider all pathophysiological mechanisms and guideline recommendations
3. Conclude with the single best answer using EXACTLY the specified format""",
        },  # Include if used in fine-tuning
        {"role": "user", "content": question},
        {"role": "assistant", "content": ""},  # Leave empty for generation
    ]

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# 1. Initialize with template verification
# model = LLM("mikkel-werling/DeepSeek-R1-Distill-Qwen-1.5B-Patient-Descriptions")
# tokenizer = AutoTokenizer.from_pretrained(
#     "mikkel-werling/DeepSeek-R1-Distill-Qwen-1.5B-Patient-Descriptions"
# )

# # 2. Verify chat template exists
# assert tokenizer.chat_template is not None, "Model missing chat template!"

# # 3. Test formatting
# test_prompt = format_with_chat_template(
#     "Hello?", "mikkel-werling/DeepSeek-R1-Distill-Qwen-1.5B-Patient-Descriptions"
# )
# print("Formatted prompt:\n", test_prompt)

# # 4. Generate with repetition safeguards
# sampling_params = anti_repetition_params(tokenizer)
# raw_output = model.generate([test_prompt], sampling_params)[0]
# print(raw_output)


# 5. Post-process using training-aware cleaning
def training_aware_cleaner(text: str) -> str:
    # Remove any template leakage from training
    return text.split(tokenizer.eos_token)[0].replace("Assistant: ", "").strip()


# final_output = training_aware_cleaner(raw_output)


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


def generate_completions(
    model: LLM,
    tokenizer,
    prompts: list[str],
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 32768,
    n: int = 1,
) -> list[str]:
    """Generate completions for a list of prompts"""
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=n,
        frequency_penalty=1.2,  # Strong repetition suppression
        presence_penalty=0.9,
        stop_token_ids=get_stop_token_ids(tokenizer),
        min_p=0.05,  # New in vLLM 0.4.2 - helps prevent degenerate outputs
    )

    start_time = time.time()
    outputs = model.generate(prompts, sampling_params)
    elapsed = time.time() - start_time

    # Extract generated text
    completions = []
    for output in outputs:
        for choice in output.outputs:
            completions.append(choice.text)

    # Clean up after inference
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Reset CUDA device to fully clear memory
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()  # Wait for all streams on the current device

    print(f"Generated {len(prompts)} prompts in {elapsed:.2f}s")
    return completions, outputs


dataframe = pd.DataFrame(dataset_test["train"])

# Example usage
if __name__ == "__main__":
    # Initialize model (replace with your HF model ID)

    models = [
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        # "mikkel-werling/DeepSeek-R1-Distill-Qwen-7B-Patient-Descriptions",
        # "mikkel-werling/DeepSeek-R1-Distill-Qwen-1.5B-Tasks",
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
        responses, outputs = generate_completions(
            model=model, tokenizer=tokenizer, prompts=specific_prompts
        )

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
