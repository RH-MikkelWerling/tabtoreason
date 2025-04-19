from prompt_functions import *
import pandas as pd

nest_asyncio.apply()  # Allows nested event loops (only needed for Jupyter)
tqdm.pandas()

biobank_processed_data = pd.read_csv("data/biobank/processed_data/biobank.csv")

biobank_processed_data_subset = biobank_processed_data[
    biobank_processed_data["patient_description_answering"].isna()
].reset_index(drop=True)

X = biobank_processed_data_subset[
    [
        x
        for x in biobank_processed_data_subset.columns
        if x != "Clinical Event Occurrence"
        and x != "Time to Clinical Event (Days)"
        and x != "index"
        and x != "patient_description_reasoning"
        and x != "patient_description_answering"
    ]
]

y = biobank_processed_data_subset["Clinical Event Occurrence"]

dictionaries = X.apply(lambda x: dict(x), axis=1)

patient_description_prompts = [
    f"{FROM_JSON_TO_QUESTION_PROMPT_BIOBANK}{json_representation}"
    for json_representation in dictionaries
]

results = asyncio.run(run_llm_calls(patient_description_prompts))  # Run tasks properly

reasoning = [x["reasoning"] for x in results]
answer = [x["answer"] for x in results]

biobank_processed_data["patient_description_reasoning"].isna().sum()

biobank_processed_data.loc[
    biobank_processed_data["patient_description_answering"].isna(),
    "patient_description_reasoning",
] = reasoning
biobank_processed_data.loc[
    biobank_processed_data["patient_description_answering"].isna(),
    "patient_description_answering",
] = answer

biobank_processed_data.to_csv(
    "data/biobank/processed_data/biobank_complete.csv", index=False
)
