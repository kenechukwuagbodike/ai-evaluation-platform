import streamlit as st
import pandas as pd
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.faithfulness import evaluate_faithfulness
from evaluation.consistency_check import evaluate_consistency

st.set_page_config(page_title="LLM Evaluation Tool", layout="wide")

st.title("🧠 AI Evaluation Platform")
st.markdown("Upload your prompts and responses CSVs, choose evaluation types, and visualize the results.")

# Sidebar
st.sidebar.header("⚙️ Configuration")
uploaded_prompts = st.sidebar.file_uploader("Upload Prompts CSV", type="csv")
uploaded_responses = st.sidebar.file_uploader("Upload Responses CSV", type="csv")

run_faithfulness = st.sidebar.checkbox("Evaluate Faithfulness")
run_consistency = st.sidebar.checkbox("Evaluate Consistency")

run_button = st.sidebar.button("▶️ Run Evaluation")

# Main Logic
if run_button:
    if uploaded_prompts is None or uploaded_responses is None:
        st.error("Please upload both 'prompts.csv' and 'responses.csv'")
    else:
        try:
            prompts_df = pd.read_csv(uploaded_prompts)
            responses_df = pd.read_csv(uploaded_responses)
            st.success("✅ Files loaded successfully.")
            st.dataframe(prompts_df.head())
            st.dataframe(responses_df.head())
        except Exception as e:
            st.error(f"❌ Failed to load CSVs: {e}")
            st.stop()

        st.markdown("## 🧪 Evaluation Results")

        # Run Faithfulness
        if run_faithfulness:
            st.markdown("### 🔍 Faithfulness Scores")
            try:
                start_time = time.time()
                faithfulness_df = evaluate_faithfulness(prompts_df, responses_df)
                st.dataframe(faithfulness_df)
                st.download_button(
                    label="Download Faithfulness Scores",
                    data=faithfulness_df.to_csv(index=False).encode("utf-8"),
                    file_name="faithfulness_scores.csv",
                    mime="text/csv"
                )
                st.success(f"✅ Faithfulness evaluation completed in {time.time() - start_time:.2f} seconds.")
            except Exception as e:
                st.error(f"❌ Error running faithfulness evaluation: {e}")

        # Run Consistency
        if run_consistency:
            st.markdown("### ♻️ Consistency Heatmap")
            start_time = time.time()
            try:
                consistency_matrix, heatmap_fig, avg_score = evaluate_consistency(responses_df)

                st.plotly_chart(heatmap_fig, use_container_width=True)
                st.metric("📊 Average Consistency Score", f"{avg_score:.2f}")

                csv_matrix = consistency_matrix.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Consistency Matrix", csv_matrix, file_name="consistency_matrix.csv")

                st.success(f"✅ Consistency evaluation completed in {time.time() - start_time:.2f} seconds.")
            except Exception as e:
                st.error(f"❌ Error running consistency evaluation:\n\n{e}")
