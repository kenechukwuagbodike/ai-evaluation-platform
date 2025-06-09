from time import sleep
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI

# Step 1: Load prompts from CSV
prompts_df = pd.read_csv("data/prompts.csv")

# Step 2: Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key = OPENAI_API_KEY)

# Step 3: Store results in a list
responses = []

# Step 4: Loop through all prompts and generate responses
for index, row in prompts_df.iterrows():
    prompt_id = row['prompt_id']
    prompt_text = row['prompt_text']
    
    try:
        response =client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.7
        )
        
        output = response.choices[0].message.content
        responses.append({'prompt_id': prompt_id, 'response_text': output})
        
        print(f"✅ Prompt {prompt_id} completed.")
        sleep(2)  # avoid rate limits

    except Exception as e:
        print(f"⚠️ Prompt {prompt_id} failed: {e}")
        
        
# Step 5: Extract and save all responses to CSV
responses_df = pd.DataFrame(responses)
responses_df.to_csv("data/responses.csv", index=False)
print("✅ All responses saved to data/responses.csv")