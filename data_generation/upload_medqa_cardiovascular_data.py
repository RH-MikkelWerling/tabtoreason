import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset

data = pd.read_csv("data/benchmarks/medqa_cardiovascular.csv")

dataset = DatasetDict({"train": Dataset.from_pandas(data)})

dataset.push_to_hub("mikkel-werling/medqa_cardiovascular")

dataset_test = load_dataset("mikkel-werling/medqa_cardiovascular")