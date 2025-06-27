from openai import OpenAI
from dotenv import load_dotenv
import os
from time import sleep

# Step 1: Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Step 2: Set the prompt to test
prompt = "What are the main concerns citizens have about the Net Zero 2050 climate goals?"

# Step 3: Generate multiple responses
responses = []
n_runs = 5

for i in range(n_runs):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7  # you can try 0.0 later to test for determinism
        )
        output = response.choices[0].message.content
        responses.append(output)
        print(f"✅ Run {i+1} complete.")
        sleep(2)
    except Exception as e:
        print(f"⚠️ Run {i+1} failed: {e}")
        responses.append("[ERROR] " + str(e))

# Step 4: Save to file
with open("data/consistency_netzero.txt", "w", encoding="utf-8") as f:
    for i, r in enumerate(responses):
        f.write(f"--- Response {i+1} ---\n{r}\n\n")

print("📄 Responses saved to data/consistency_netzero.txt")
