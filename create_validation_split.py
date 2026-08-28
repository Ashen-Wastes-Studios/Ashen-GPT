import os

input_filename = "train_split.txt"
val_filename = "val_split.txt"

if not os.path.exists(input_filename):
    print(f"Error: {input_filename} not found.")
    exit(1)

print(f"Reading {input_filename}...")
with open(input_filename, "r", encoding="utf-8") as f:
    text = f.read()

total_chars = len(text)
print(f"Total corpus size: {total_chars / (1024*1024):.2f} MB ({total_chars:,} characters)")

# Split ratio: 90% train, 10% validation
split_index = int(0.9 * total_chars)

train_text = text[:split_index]
val_text = text[split_index:]

print(f"Writing training split ({len(train_text):,} chars) to {input_filename}...")
with open(input_filename, "w", encoding="utf-8") as f:
    f.write(train_text)

print(f"Writing validation split ({len(val_text):,} chars) to {val_filename}...")
with open(val_filename, "w", encoding="utf-8") as f:
    f.write(val_text)

print("Validation split successfully created!")
