import urllib.request
import re
import os
import time

output_filename = "train_split.txt"

print(f"Starting curated Project Gutenberg downloader...", flush=True)
print(f"Output file: {output_filename}\n", flush=True)

# Curated list of hundreds of known valid classic book IDs across Project Gutenberg
KNOWN_BOOK_IDS = [
    11, 84, 98, 135, 174, 345, 430, 520, 613, 768, 844, 932, 1001, 1080, 1100, 
    1200, 1232, 1342, 1400, 1500, 1661, 1952, 2000, 2500, 2542, 2600, 2701, 
    3000, 3497, 4300, 5200, 6130, 7178, 7400, 768, 996, 120, 160, 209, 219,
    # Adding more known ranges/IDs
    *range(3001, 3500, 5),
    *range(4000, 4500, 5),
    *range(5000, 5500, 5),
    *range(10000, 11000, 5),
    *range(15000, 16000, 5),
    *range(20000, 21000, 5),
]

success_count = 0

for book_id in KNOWN_BOOK_IDS:
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    print(f"Downloading Book ID {book_id}...", flush=True)
    
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) ResearchBot/1.0'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            text = response.read().decode('utf-8', errors='ignore')
            
            # Simple cleaning of Gutenberg header/footer
            start_match = re.search(r"\*\*\* START OF THE (?:PROJECT GUTENBERG|EBOOK).*?\*\*\*", text, re.IGNORECASE)
            end_match = re.search(r"\*\*\* END OF THE (?:PROJECT GUTENBERG|EBOOK).*?\*\*\*", text, re.IGNORECASE)
            
            if start_match and end_match:
                text = text[start_match.end():end_match.start()]
            
            cleaned_text = f"\n\n--- BOOK ID: {book_id} ---\n\n" + text.strip() + "\n"
            
            # Append immediately to file
            with open(output_filename, "a", encoding="utf-8") as f:
                f.write(cleaned_text)
            
            success_count += 1
            current_size = os.path.getsize(output_filename) if os.path.exists(output_filename) else 0
            print(f"==> [SUCCESS] Book ID {book_id} saved! Total downloaded: {success_count} | File size: {current_size / (1024*1024):.2f} MB", flush=True)
                
        time.sleep(0.3)
        
    except Exception as e:
        print(f"[-] Book ID {book_id} not found or error: {e}", flush=True)
        time.sleep(0.1)
        continue

print(f"\nCurated download complete! Total books downloaded: {success_count}. Saved to {output_filename}.", flush=True)
