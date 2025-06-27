# Step 1: Load raw responses from file
file_path = "data/consistency_netzero.txt"

with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

# Step 2: Split responses using the delimiter
responses = raw_text.split("--- Response")[1:]  # skip the first empty split
responses = [r.strip().split("\n", 1)[1].strip() for r in responses if "\n" in r]  # extract text after header

# Step 3: Print how many we loaded
print(f"✅ Loaded {len(responses)} responses")
print("\n🧾 First response:\n")
print(responses[0][:400] + "...\n")  # show a preview
