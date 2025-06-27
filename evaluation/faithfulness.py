from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd

def evaluate_faithfulness(prompt_df, response_df):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    faithfulness_scores = []

    for _, row in response_df.iterrows():
        prompt_id = row["prompt_id"]
        prompt_text = prompt_df[prompt_df["prompt_id"] == prompt_id]["prompt_text"].values[0]
        response_text = row["response_text"]

        embeddings = model.encode([prompt_text, response_text])
        score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

        faithfulness_scores.append({"prompt_id": prompt_id, "faithfulness_score": score})

    return pd.DataFrame(faithfulness_scores)
