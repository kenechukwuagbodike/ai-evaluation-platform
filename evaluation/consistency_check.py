from sentence_transformers import SentenceTransformer
import os
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px

# Function to evaluate consistency of responses
# This function computes the cosine similarity between all pairs of responses


def evaluate_consistency(responses_df):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Get responses and encode them
    responses = responses_df["response_text"].tolist()
    embeddings = model.encode(responses)

    # Cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(similarity_matrix, np.nan)  # Ignore self-similarity in avg

    avg_score = np.nanmean(similarity_matrix)

    # Create DataFrame for Plotly
    matrix_df = pd.DataFrame(similarity_matrix, columns=[f"R{i}" for i in range(len(responses))], index=[f"R{i}" for i in range(len(responses))])

    # Create interactive heatmap
    fig = px.imshow(
        matrix_df,
        text_auto=".2f",
        color_continuous_scale="YlGnBu",
        labels={"color": "Cosine Similarity"},
        title="🔄 Consistency Similarity Heatmap"
    )

    fig.update_layout(
        width=600,
        height=600,
        margin=dict(t=40, l=0, r=0, b=0)
    )

    return matrix_df, fig, avg_score

# Step 1: Load the raw GPT-4 responses
if __name__ == "__main__":
    # Step 1: Load the raw GPT-4 responses
    file_path = "data/consistency_netzero.txt"

    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Step 2: Split responses into a list
    responses = raw_text.split("--- Response")[1:]  # ignore the empty first part
    responses = [r.strip().split("\n", 1)[1].strip() for r in responses if "\n" in r]  # get text after header

    print(f"✅ Loaded {len(responses)} responses")
    print("\n🧾 Sample (response 1):")
    print(responses[0][:300] + "...\n")

    # Step 3: Load the embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Step 4: Encode all responses into embeddings
    embeddings = model.encode(responses)
    print(f"🧠 Embeddings shape: {embeddings.shape}")

    # Step 5: Compute pairwise similarity
    similarity_matrix = cosine_similarity(embeddings)

    # Step 6: Print the raw matrix
    print("\n🔁 Pairwise Similarity Matrix:")
    print(np.round(similarity_matrix, 2))

    # Step 7: Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_matrix, annot=True, cmap="Blues", vmin=0, vmax=1, fmt=".2f", annot_kws={"size": 8}, linewidths=0.5,
                xticklabels=[f"R{i+1}" for i in range(len(responses))],
                yticklabels=[f"R{i+1}" for i in range(len(responses))])
    plt.title("GPT-4 Response Consistency Heatmap")
    plt.tight_layout()
    plt.savefig("data/consistency_heatmap.png")
    print("📊 Consistency heatmap saved as data/consistency_heatmap.png")

    # Compute average off-diagonal similarity (consistency score)
    n = len(similarity_matrix)
    off_diagonal_scores = similarity_matrix[np.triu_indices(n, k=1)]
    consistency_score = round(off_diagonal_scores.mean(), 4)

    print(f"\n📈 Average Consistency Score (off-diagonal): {consistency_score}")
