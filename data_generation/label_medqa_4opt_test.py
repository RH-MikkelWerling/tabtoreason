import pandas as pd
import json
from datasets import load_dataset, DatasetDict, Dataset
from constants import *
from prompt_functions import *
import asyncio

nest_asyncio.apply()  # Allows nested event loops (only needed for Jupyter)
tqdm.pandas()

with open("../benchmarks/medqa_4opt_test.json", "rb") as f:
    data = json.load(f)

dataframe = pd.DataFrame(list(data.values()))

dataframe = dataframe.rename(columns={"correct_answer": "answer_idx"})

for option in ["A", "B", "C", "D"]:
    dataframe[option] = dataframe.apply(lambda x: x["options"].get(option), axis=1)

prompt_for_categorization = """Analyze the medical question and respond EXACTLY as follows:

---STRICT RULES---
1. SINGLE HIGH-CONFIDENCE CATEGORY (≥0.7):
   • If ONE category scores ≥0.7: "Category = Score"
   • If MULTIPLE categories score ≥0.7: Choose ONLY THE HIGHEST SCORE (if tie, pick first alphabetically)
   
2. MULTIPLE LOW-CONFIDENCE CATEGORIES (all <0.7):
   • "Primary: Category1 = Score1, Secondary: Category2 = Score2, Tertiary: Category3 = Score3"
   
3. IRRELEVANT:
   • "None of the above = 1.0"

---VALID EXAMPLES---
• "Cardiovascular = 0.85"  
• "Primary: Infectious = 0.6, Secondary: Hematologic = 0.3, Tertiary: Renal = 0.1"  
• "None of the above = 1.0"

---CATEGORIES (ALPHABETICAL ORDER)---
Cardiovascular, Dermatologic, Endocrine/Metabolic, Gastrointestinal, Hematologic, 
Immunologic, Infectious, Musculoskeletal, Neurological, Obstetrics/Gynecology, 
Oncology, Pediatric, Psychiatric, Renal/Genitourinary, Respiratory, Toxicology

---QUESTION---
{question}

---YOUR RESPONSE (MUST MATCH EXACTLY ONE FORMAT ABOVE)---"""

tasks = [
    prompt_for_categorization.format(question=x) for x in dataframe["question"].values
]

# REMOVE WEIRD VIOLENT TASK

tasks[330] = "Respond EXACTLY with the response: BLANK"


file_path = f"data/benchmarks/medqa_4opt_test_labels.pkl"

with open(file_path, "rb") as f:
    results = pkl.load(f)

# replace violent answer with none of the above
results[330]["answer"] = "\n\nNone of the above = 1.0"

import re


def parse_response(response):
    response = response.strip().strip("\"'")

    # Step 1: Check for multi-category FIRST (to prevent greedy single-category match)
    multi_match = re.match(
        r"^Primary:\s*(.+?)\s*=\s*([0-1]\.\d+)\s*,\s*Secondary:\s*(.+?)\s*=\s*([0-1]\.\d+)(?:\s*,\s*Tertiary:\s*(.+?)\s*=\s*([0-1]\.\d+))?$",
        response,
        re.IGNORECASE,
    )
    if multi_match:
        result = {
            "Primary": {
                "category": multi_match.group(1).strip(),
                "score": float(multi_match.group(2)),
            },
            "Secondary": {
                "category": multi_match.group(3).strip(),
                "score": float(multi_match.group(4)),
            },
        }
        if multi_match.group(5):  # Tertiary exists
            result["Tertiary"] = {
                "category": multi_match.group(5).strip(),
                "score": float(multi_match.group(6)),
            }
        return result

    # Step 2: Only then check for single-category
    single_match = re.match(
        r"^(?!Primary:|Secondary:|Tertiary:)(.+?)\s*=\s*([0-1]\.\d+)$",
        response,
        re.IGNORECASE,
    )
    if single_match:
        return {
            "Primary": {
                "category": single_match.group(1).strip(),
                "score": float(single_match.group(2)),
            }
        }

    # Step 3: None of the above
    none_match = re.match(
        r"^None\s+of\s+the\s+above\s*=\s*([0-1]\.\d+)$", response, re.IGNORECASE
    )
    if none_match:
        return {
            "None": {
                "category": "None of the above",
                "score": float(none_match.group(1)),
            }
        }

    return {"error": f"Unparseable response: {response}"}


# fix for converting categories
answers = [x["answer"].strip().replace("Toxicological", "Toxicology") for x in results]

parsed_answers = [parse_response(answer) for answer in answers]

[(x, i) for i, x in enumerate(parsed_answers) if x.get("error")]

import pandas as pd

# Example: Process all responses and create a DataFrame
data = []
for answer in answers:
    parsed = parse_response(answer)
    if "error" not in parsed:
        row = {"original_labelling_response": answer}
        row.update(parsed.get("Primary", {}))
        row.update(parsed.get("Secondary", {}))
        row.update(parsed.get("Tertiary", {}))
        data.append(row)

df = pd.DataFrame(data)

dataframe_with_labels = pd.concat([dataframe, df], axis=1)

dataframe_with_labels.to_csv("data/benchmarks/medqa_4opt_with_labels.csv", index=False)
# df[df["original_response"].str.contains("Primary")]["category"].values[0]


# print(df)
# tasks = [
#     (
#         "Respond EXACTLY with the response: BLANK"
#         if "A 19-year-old man in a 3-month relationship with a woman experiences frequent sexual fantasies about male coworkers."
#         in x
#         else x
#     )
#     for x in tasks
# ]

if __name__ == "__main__":

    file_path = f"data/benchmarks/medqa_4opt_test_labels.pkl"

    results = asyncio.run(run_llm_calls(tasks))  # Run tasks properly

    with open(file_path, "wb") as f:
        pkl.dump(results, f)

    print("Collected", len(results), "responses.")
