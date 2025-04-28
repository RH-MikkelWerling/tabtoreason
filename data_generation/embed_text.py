from constants import *
from openai import OpenAI, AzureOpenAI, AsyncAzureOpenAI
import pandas as pd
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
import asyncio
import nest_asyncio
import math
import random

nest_asyncio.apply()  # Allows nested event loops (only needed for Jupyter)

tqdm.pandas()

# read data

biobank_processed_data = pd.read_csv("data/biobank/processed_data/biobank.csv")

# make client
client = AsyncAzureOpenAI(
    azure_endpoint=EMBEDDING_ENDPOINT,
    api_key=EMBEDDING_API_KEY,
    api_version="2023-09-01-preview",
)


def is_valid_embedding(embedding):
    return all(
        isinstance(x, float) and not math.isnan(x) and not math.isinf(x)
        for x in embedding
    )


async def generate_embedding_with_retry(text: str, semaphore, delay, max_retries=5):
    async with semaphore:
        for attempt in range(1, max_retries + 1):
            try:
                await asyncio.sleep(delay)  # spacing for rate limits

                response = await client.embeddings.create(
                    input=text, model="text-embedding-3-large"
                )
                embedding = response.data[0].embedding

                if is_valid_embedding(embedding):
                    return embedding
                else:
                    raise ValueError("Invalid embedding: contains NaN or inf")

            except Exception as e:
                wait_time = delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                print(
                    f"⚠️ Retry {attempt}/{max_retries} for text: {e}. Waiting {wait_time:.2f}s..."
                )
                await asyncio.sleep(wait_time)

        print(f"❌ Failed after {max_retries} attempts.")
        return None  # or raise, if you prefer


# async function to create embedding
async def generate_embedding_from_text(text: str):
    response = await client.embeddings.create(
        input=text, model="text-embedding-3-large"
    )
    return response.data[0].embedding


async def generate_all_embeddings_with_retry(texts, max_requests_per_minute=60):
    delay_between_requests = 60 / max_requests_per_minute
    semaphore = asyncio.Semaphore(10)

    tasks = [
        generate_embedding_with_retry(text, semaphore, delay_between_requests)
        for text in texts
    ]

    embeddings = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        embedding = await coro
        embeddings.append(embedding)

    return embeddings


# assuming biobank_processed_data is a pandas DataFrame
texts = biobank_processed_data["patient_description_answering"].tolist()[:100]

# run the async tasks
embeddings = asyncio.run(generate_all_embeddings_with_retry(texts))

embeddings

# save back into the DataFrame
biobank_processed_data["embeddings"] = embeddings

# assuming biobank_processed_data is a pandas DataFrame
texts = biobank_processed_data[biobank_processed_data["embeddings"].isna()][
    "patient_description_answering"
].tolist()

# run the async tasks
embeddings_new = asyncio.run(generate_all_embeddings_with_retry(texts))

second_iter = iter(embeddings_new)
embeddings_all = [emb if emb is not None else next(second_iter) for emb in embeddings]

np.save("data/biobank/processed_data/embeddings.npy", np.array(embeddings_all))


biobank_processed_data["embeddings_all"] = embeddings_all

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Convert embeddings to array
embedding_matrix = np.array(biobank_processed_data["embeddings_all"].tolist())

# Step 2: Normalize (cosine similarity = dot product of unit vectors)
# embedding_matrix = normalize(embedding_matrix)

# Step 2: Masks and split
outcome = biobank_processed_data["Clinical Event Occurrence"].values
is_alive = outcome == 1
is_dead = outcome == 0

embeddings_alive = embedding_matrix[is_alive]
embeddings_dead = embedding_matrix[is_dead]

index_alive = biobank_processed_data[is_alive].index.to_numpy()
index_dead = biobank_processed_data[is_dead].index.to_numpy()

# Step 3: Compute cosine similarities in bulk
similarity_to_alive = embedding_matrix @ embeddings_alive.T  # shape: (n, n_alive)
similarity_to_dead = embedding_matrix @ embeddings_dead.T  # shape: (n, n_dead)

# Step 4: Exclude self-matches if in same group
for i in tqdm(range(len(embedding_matrix))):
    if is_alive[i]:
        # This person is alive, might be in the "alive" comparison set
        alive_idx = np.where(index_alive == biobank_processed_data.index[i])[0]
        if alive_idx.size > 0:
            similarity_to_alive[i, alive_idx[0]] = -1
    if is_dead[i]:
        dead_idx = np.where(index_dead == biobank_processed_data.index[i])[0]
        if dead_idx.size > 0:
            similarity_to_dead[i, dead_idx[0]] = -1

# Step 5: Get best match index and similarity
best_alive_idx = np.argmax(similarity_to_alive, axis=1)
best_dead_idx = np.argmax(similarity_to_dead, axis=1)

closest_alive_ids = index_alive[best_alive_idx]
closest_dead_ids = index_dead[best_dead_idx]

closest_alive_sims = similarity_to_alive[np.arange(len(embeddings)), best_alive_idx]
closest_dead_sims = similarity_to_dead[np.arange(len(embeddings)), best_dead_idx]

# Step 6: Store in DataFrame
biobank_processed_data["closest_alive_id"] = closest_alive_ids
biobank_processed_data["closest_alive_similarity"] = closest_alive_sims

biobank_processed_data["closest_dead_id"] = closest_dead_ids
biobank_processed_data["closest_dead_similarity"] = closest_dead_sims

# lets just try to plot it for fun


biobank_processed_data.to_csv("data/biobank/processed_data/biobank_with_embeddings.csv")

descriptions = biobank_processed_data["patient_description_answering"].values


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap


def plot_umap_embeddings(
    df, embedding_col="embeddings", color_by="outcome", title=None, cmap="viridis"
):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import umap

    # Extract and reduce embeddings
    embeddings = np.vstack(df[embedding_col].values)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    umap_coords = reducer.fit_transform(embeddings)

    # Prepare for plotting
    df_plot = df.copy()
    df_plot["umap_x"] = umap_coords[:, 0]
    df_plot["umap_y"] = umap_coords[:, 1]

    # Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x="umap_x",
        y="umap_y",
        hue=color_by,
        data=df_plot,
        palette=cmap,
        s=4,  # smaller points
        alpha=0.6,  # more transparency
        linewidth=0.1,
        edgecolor="k",  # slight outline
    )

    plt.title(title or f"UMAP of Embeddings Colored by {color_by}")
    plt.axis("off")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def make_label_specific_columns(row):
    if row["Clinical Event Occurrence"] == 0:
        same_label_closest_distance = row["closest_alive_similarity"]
        other_label_closest_distance = row["closest_dead_similarity"]
    else:
        same_label_closest_distance = row["closest_dead_similarity"]
        other_label_closest_distance = row["closest_alive_similarity"]
    return same_label_closest_distance, other_label_closest_distance


biobank_processed_data[
    ["closest_distance_same_label", "closest_distance_other_label"]
] = biobank_processed_data.apply(
    lambda x: make_label_specific_columns(x), axis=1, result_type="expand"
)


def calculate_difficulty_of_task(row, alpha=0.5):
    difficulty = (
        alpha * row["closest_distance_other_label"]
        + (1 - alpha) * row["closest_distance_same_label"]
    )
    return difficulty


def calculate_difficulty_of_task(row):
    difficulty = (row["closest_distance_same_label"]) / (
        row["closest_distance_other_label"]
    )
    return 1 / (1 + np.exp(-difficulty))


biobank_processed_data["difficulty"] = biobank_processed_data.apply(
    lambda x: calculate_difficulty_of_task(x), axis=1
)


plot_umap_embeddings(
    biobank_processed_data,
    embedding_col="embeddings_all",
    color_by="Clinical Event Occurrence",
    cmap="coolwarm",
)

plot_umap_embeddings(
    biobank_processed_data,
    embedding_col="embeddings_all",
    color_by="Age (Years)",
    cmap="plasma",
)


plot_umap_embeddings(
    biobank_processed_data,
    embedding_col="embeddings_all",
    color_by="closest_alive_similarity",
    cmap="plasma",
)


plot_umap_embeddings(
    biobank_processed_data,
    embedding_col="embeddings_all",
    color_by="closest_dead_similarity",
    cmap="plasma",
)


sns.displot(data=biobank_processed_data, x="difficulty", kde=True)

plot_umap_embeddings(
    biobank_processed_data,
    embedding_col="embeddings_all",
    color_by="difficulty",
    cmap="plasma",
)
