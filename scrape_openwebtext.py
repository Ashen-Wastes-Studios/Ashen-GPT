from datasets import load_dataset
import os

output_filename = "train_split.txt"

print("Loading OpenWebText in streaming mode from Hugging Face...")
dataset = load_dataset("Skylion007/openwebtext", streaming=True)

current_size = os.path.getsize(output_filename) if os.path.exists(output_filename) else 0
print(f"Current training file size: {current_size / (1024*1024):.2f} MB")

print("Streaming and appending OpenWebText documents to train_split.txt...")
count = 0

try:
    with open(output_filename, "a", encoding="utf-8") as f:
        for sample in dataset['train']:
            text = sample['text'].strip()
            if len(text) > 200:  # Skip very short snippets
                f.write(text + "\n\n<|endoftext|>\n\n")
                count += 1
                
                if count % 100 == 0:
                    current_size = os.path.getsize(output_filename)
                    print(f"Appended {count} OpenWebText documents. Current file size: {current_size / (1024*1024*1024):.2f} GB", flush=True)
                    
                # Optional: set a limit per run, e.g., 50,000 documents
                if count >= 50000:
                    print("Reached 50,000 OpenWebText documents for this batch.")
                    break
except KeyboardInterrupt:
    print(f"\nStopped by user. Total OpenWebText documents appended: {count}")

print(f"Done! Successfully added OpenWebText data to {output_filename}.")
