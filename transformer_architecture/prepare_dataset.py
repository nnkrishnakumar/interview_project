import pandas as pd
import os
from sklearn.model_selection import train_test_split

# CONFIG
INPUT_CSV = r"C:\Users\aimar\OneDrive\Desktop\project_interview\transformer_architecture\samples\fairytale_data.csv"  # Change to your actual file name
OUTPUT_DIR = "data"
VAL_SPLIT = 0.1  # 10% for validation

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load CSV
df = pd.read_csv(INPUT_CSV)

# Combine fields into training lines
def format_row(row):
    return f"Context: {row['context']} Question: {row['question']} Answer: {row['answer']}"

texts = df.apply(format_row, axis=1).tolist()

# Split
train_texts, val_texts = train_test_split(texts, test_size=VAL_SPLIT, random_state=42)

# Write to files
with open(os.path.join(OUTPUT_DIR, "train.txt"), "w", encoding="utf-8") as f:
    for line in train_texts:
        f.write(line.strip() + "\n")

with open(os.path.join(OUTPUT_DIR, "val.txt"), "w", encoding="utf-8") as f:
    for line in val_texts:
        f.write(line.strip() + "\n")

print(f"Saved {len(train_texts)} training samples and {len(val_texts)} validation samples.")
