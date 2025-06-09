import pandas as pd

# Step 1: Load responses from CSV
responses_df = pd.read_csv("data/responses.csv")

# Step 2: Show how many rows we have
num_rows = len(responses_df)
print(f"Total responses loaded: {num_rows}")

#Step 3: Display a preview
print("Preview of responses:")
print(responses_df.head())

# Step 4: Check for missing values
missing_values = responses_df['response_text'].str.contains('Error|NaN').sum()