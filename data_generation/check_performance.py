import pandas as pd

base_model = pd.read_csv(
    "data/biobank/processed_data/DeepSeek-R1-Distill-Qwen-1.5B_responses.csv"
)

base_model["correct_answer"] = (
    base_model["extracted_answers"] == base_model["answer_idx"]
)

base_model["correct_answer"].sum() / len(base_model)

finetuned_model = pd.read_csv(
    "data/biobank/processed_data/DeepSeek-R1-Distill-Qwen-1.5B-Patient-Descriptions_responses.csv"
)


finetuned_model = pd.read_csv(
    "data/biobank/processed_data/DeepSeek-R1-Distill-Llama-8B_responses.csv"
)


finetuned_model["correct_answer"] = (
    finetuned_model["extracted_answers"] == finetuned_model["answer_idx"]
)

finetuned_model["correct_answer"].sum() / len(finetuned_model)

finetuned_tasks_model = pd.read_csv(
    "data/biobank/processed_data/DeepSeek-R1-Distill-Qwen-7B-Tasks_responses.csv"
)

import re


def extract_answer(response) -> str:
    """Robust answer extraction with group validation"""
    response = str(response) if pd.notna(response) else ""

    patterns = [
        # Group 1 capture patterns
        (r"(?:Answer|Final\s+Determination)[:\s]*([A-D])\)?", 1),  # Pattern 1
        (r"\bOption\s*([A-D])\)", 1),  # Pattern 2
        (r"\n([A-D])[\s\.]*$", 1),  # Pattern 3
        (r"(?<![a-zA-Z])([A-D])(?![a-zA-Z])", 1),  # Pattern 4 (now with capture group)
    ]

    for pattern, group in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match and match.lastindex >= group:
            return match.group(group).upper()

    return None  # No valid match found


# Handle NaN/float responses in your dataset
base_model["clean_response"] = base_model["model_answer"].fillna("")
base_model["extracted"] = base_model["clean_response"].apply(extract_answer)

base_model["extracted"].isna().sum()

base_model["correct_answer"] = base_model["answer_idx"] == base_model["extracted"]

base_model["correct_answer"].sum() / len(base_model)

# Handle NaN/float responses in your dataset
finetuned_model["clean_response"] = finetuned_model["model_answer"].fillna("")
finetuned_model["extracted"] = finetuned_model["clean_response"].apply(extract_answer)

finetuned_model["extracted"].isna().sum()

finetuned_model[finetuned_model["extracted"].isna()]["clean_response"].values

finetuned_model["correct_answer"] = (
    finetuned_model["answer_idx"] == finetuned_model["extracted"]
)

finetuned_model["correct_answer"].sum() / len(finetuned_model)

finetuned_tasks_model["clean_response"] = finetuned_tasks_model["model_answer"].fillna(
    ""
)
finetuned_tasks_model["extracted"] = finetuned_tasks_model["clean_response"].apply(
    extract_answer
)

finetuned_tasks_model["extracted"].isna().sum()

finetuned_tasks_model[finetuned_tasks_model["extracted"].isna()][
    "clean_response"
].values[2]

finetuned_tasks_model[finetuned_tasks_model["extracted"].isna()]["model_answer"].values[
    -1
]


finetuned_tasks_model["correct_answer"] = (
    finetuned_tasks_model["answer_idx"] == finetuned_tasks_model["extracted"]
)

finetuned_tasks_model["correct_answer"].sum() / len(finetuned_tasks_model)
