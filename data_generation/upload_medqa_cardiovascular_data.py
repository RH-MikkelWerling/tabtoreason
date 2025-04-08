import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset

data = pd.read_csv("data/benchmarks/medqa_cardiovascular.csv")

dataset = DatasetDict({"train": Dataset.from_pandas(data)})

dataset.push_to_hub("mikkel-werling/medqa_cardiovascular")

dataset_test = load_dataset("mikkel-werling/medqa_cardiovascular")

dataset_test

dataset_test = load_dataset("open-r1/OpenR1-Math-220k")

dataset_test

print(dataset_test["train"].map())

# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
#     src/open_r1/sft.py \
#     --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/sft/config_cardiovascular_biobank.yaml
