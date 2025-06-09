# AI Evaluation Platform for Public Consultation Responses

## Overview

This platform is designed to evaluate the outputs of Large Language Models (LLMs) in the context of public policy consultations. It focuses on assessing:

- **Faithfulness**: Accuracy of the summary in representing the original input.
- **Fairness/Bias**: Detection of any skewness towards particular viewpoints or demographics.
- **Robustness**: Consistency of responses across different runs or models.
- **Explainability**: Clarity in the reasoning behind the model's outputs.

## Features

- Modular evaluation pipeline.
- Support for multiple evaluation metrics.
- Integration with OpenAI API for generating model responses.
- Streamlit-based dashboard for interactive analysis.

## Project Structure
├/ai-eval-platform/
│
├── data/
│   ├── prompts.csv         # List of consultation-style prompts
│   ├── responses.csv       # Model-generated responses
│
├── evaluation/
│   ├── evaluation_pipeline.py   # Core logic: scoring, metrics, fairness
│   ├── report_generator.py      # Outputs visuals + summary
│
├── ui/
│   ├── streamlit_app.py    # (Optional UI)
│
├── README.md
├── requirements.txt
└── .env                    # For OpenAI keys