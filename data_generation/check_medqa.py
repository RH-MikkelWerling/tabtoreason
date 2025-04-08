import pandas as pd

from datasets import load_dataset

datasets = load_dataset("bigbio/med_qa", 'med_qa_en_4options_bigbio_qa')

datasets["train"]["choices"][0]
[x for x in dataset]

line = datasets["train"][0]

gold_index = [index for index, element in enumerate(line["choices"]) if element == line["answer"][0]][0]

gold_index

[index for index, element in enumerate(line["choices"])]

print(datasets)

gpqa = load_dataset("Idavidrein/gpqa", "gpqa_diamond")

gpqa["train"]["Correct Answer"]