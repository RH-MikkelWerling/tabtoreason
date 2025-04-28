import pandas as pd

base_model = pd.read_csv(
    "data/biobank/processed_data/DeepSeek-R1-Distill-Qwen-1.5B_responses.csv"
)

finetuned_model = pd.read_csv(
    "data/biobank/processed_data/DeepSeek-R1-Distill-Qwen-1.5B-Patient-Descriptions_responses.csv"
)

finetuned_model["model_answer"][0]

base_model["outputs"][0]


finetuned_tasks_model = pd.read_csv(
    "data/biobank/processed_data/DeepSeek-R1-Distill-Qwen-1.5B-Tasks_responses.csv"
)

base_model["model_reasoning"]
