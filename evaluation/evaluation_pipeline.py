import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Load data
    df = pd.read_csv("data/responses.csv")

    # Join prompt_text into the responses dataframe
    prompts = pd.read_csv("data/prompts.csv")
    df = df.merge(prompts, on="prompt_id", how="left")

    # Load model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Encode actual prompt_text and response_text
    prompt_embeddings = model.encode(df["prompt_text"].astype(str))
    response_embeddings = model.encode(df["response_text"].astype(str))

    # Calculate similarity per row
    similarities = []
    for i in range(len(df)):
        sim = cosine_similarity([prompt_embeddings[i]], [response_embeddings[i]])[0][0]
        similarities.append(sim)

    # Add results to dataframe
    df["faithfulness_score"] = similarities

    # Show results
    print("\n📊 Faithfulness Scores:")
    print(df[["prompt_id", "faithfulness_score"]])

    # Optional: Save for later use
    df.to_csv("data/evaluation_results.csv", index=False)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.bar(df["prompt_id"].astype(str), df["faithfulness_score"], color="steelblue")
    plt.ylim(0, 1)
    plt.xlabel("Prompt ID")
    plt.ylabel("Faithfulness Score")
    plt.title("LLM Response Faithfulness by Prompt")
    plt.axhline(0.7, color="red", linestyle="--", label="Threshold (0.7)")
    plt.legend()
    plt.tight_layout()

    # Save chart to file
    plt.savefig("data/faithfulness_scores.png")
    print("📊 Chart saved as data/faithfulness_scores.png")