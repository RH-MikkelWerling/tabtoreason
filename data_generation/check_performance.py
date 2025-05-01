import pandas as pd
import glob

data_paths = glob.glob("data/biobank/processed_data/DeepSeek-R1*")

list_of_dataframes = []
for data_path in data_paths:
    data = pd.read_csv(data_path)
    data["model"] = data_path.split("/")[-1]
    list_of_dataframes.append(data)

data = pd.concat(list_of_dataframes).reset_index(drop=True)

data = pd.read_csv(
    "data/biobank/processed_data/DeepSeek-R1-Distill-Llama-8B-Tasks_responses_new_settings.csv"
)

data["correct_answer"] = data["extracted_answers"] == data["answer_idx"]
data["model"] = "DeepSeek-R1-Distill-Llama-8B-Tasks"


def make_identifier_columns(string):
    if "8B" in string:
        size = "8B"
    elif "1.5B" in string:
        size = "1.5B"
    elif "7B" in string:
        size = "7B"
    if "Tasks" in string:
        model_type = "Counterfactuals"
    elif "Patient-Descriptions" in string:
        model_type = "Patient Descriptions"
    else:
        model_type = "Base"

    return size, model_type


data["Model Size"], data["Model Type"] = zip(
    *data["model"].apply(make_identifier_columns)
)

import seaborn as sns


sns.barplot(data=data, x="Model Size", y="correct_answer", hue="Model Type")


performance_data = (
    data.groupby(["Model Size", "Model Type"])
    .agg(mean_accuracy=("correct_answer", "mean"))
    .reset_index()
)

performance_data = (
    data.groupby(["Model Size", "Model Type", "run"])
    .agg(mean_accuracy=("correct_answer", "mean"))
    .reset_index()
)

sns.stripplot(
    data=performance_data,
    x="Model Size",
    y="mean_accuracy",
    hue="Model Type",
    dodge=True,
    size=4,
    edgecolor="black",
    linewidth=1,
    legend=False,
    alpha=0.3,
)


sns.boxplot(
    data=performance_data,
    x="Model Size",
    y="mean_accuracy",
    hue="Model Type",
    dodge=True,
    # size=5,
    # edgecolor="black",
    # linewidth=1
)
