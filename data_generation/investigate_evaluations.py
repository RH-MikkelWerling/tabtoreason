import pandas as pd
import numpy as np

data = pd.read_parquet(
    "../data/evals/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/details/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/2025-04-23T13-01-02.868871/details_helm|med_qa_4opt_cardiovascular|0_2025-04-23T13-01-02.868871.parquet"
)


def decode_prediction(row):
    # Get logits for each option
    logits = np.array(
        [x[0] for x in row["predictions"]]
    )  # Extract first value from each array

    # Convert logits to probabilities (softmax)
    probs = np.exp(logits) / np.sum(np.exp(logits))

    # Get predicted answer (index with highest probability)
    predicted_idx = np.argmax(probs)

    # Map to actual choice (A/B/C/D etc.)
    return {
        "predicted_choice": row["choices"][predicted_idx],
        "confidence": probs[predicted_idx],
        "all_probs": dict(zip(row["choices"], probs)),
    }


# Apply decoding to each row
decoded = data.apply(decode_prediction, axis=1, result_type="expand")
df = pd.concat([data, decoded], axis=1)


data_finetuned = pd.read_parquet(
    "../data/evals/mikkel-werling/DeepSeek-R1-Distill-Qwen-1.5B/details/mikkel-werling/DeepSeek-R1-Distill-Qwen-1.5B/2025-04-23T13-03-18.256095/details_helm|med_qa_4opt_cardiovascular|0_2025-04-23T13-03-18.256095.parquet"
)

# Apply decoding to each row
decoded_finetuned = data_finetuned.apply(
    decode_prediction, axis=1, result_type="expand"
)
df_finetuned = pd.concat([data_finetuned, decoded_finetuned], axis=1)

df_finetuned["predicted_choice"].value_counts()

df["predicted_choice"].value_counts()


import seaborn as sns

convertion_dict = {
    "A": np.array([0]),
    "B": np.array([1]),
    "C": np.array([2]),
    "D": np.array([3]),
}

df["predicted_golden_index"] = df["predicted_choice"].apply(
    lambda x: convertion_dict.get(x)
)

df.loc[df["gold_index"] == df["predicted_golden_index"], "correct_answer"] = "Yes"
df.loc[df["gold_index"] != df["predicted_golden_index"], "correct_answer"] = "No"


df_finetuned["predicted_golden_index"] = df_finetuned["predicted_choice"].apply(
    lambda x: convertion_dict.get(x)
)

df_finetuned.loc[
    df_finetuned["gold_index"] == df_finetuned["predicted_golden_index"],
    "correct_answer",
] = "Yes"
df_finetuned.loc[
    df_finetuned["gold_index"] != df_finetuned["predicted_golden_index"],
    "correct_answer",
] = "No"


sns.displot(data=df, x="confidence", hue="correct_answer", kde=True)

sns.displot(data=df_finetuned, x="confidence", hue="correct_answer", kde=True)


data.columns

data["predictions"]

data["predictions"].values[0]
