import urllib.request
import json
import os
import time

output_filename = "code_train_split.txt"

print(f"Starting round-robin multi-language GitHub code scraper...", flush=True)
print(f"Output file: {output_filename}\n", flush=True)

# Repositories categorized by language for round-robin interleaving
LANGUAGE_REPOS = {
    "Python": [
        ("pallets", "flask", "main", ".py"),
        ("psf", "requests", "main", ".py"),
        ("tiangolo", "fastapi", "master", ".py"),
        ("pydantic", "pydantic", "main", ".py"),
        ("django", "django", "main", ".py"),
    ],
    "JavaScript/TypeScript": [
        ("expressjs", "express", "master", (".js", ".ts")),
        ("axios", "axios", "master", (".js", ".ts")),
        ("vuejs", "core", "main", (".js", ".ts")),
    ],
    "Go": [
        ("gin-gonic", "gin", "master", ".go"),
        ("gofiber", "fiber", "master", ".go"),
    ],
    "Rust": [
        ("tokio-rs", "tokio", "master", ".rs"),
        ("serde-rs", "serde", "master", ".rs"),
    ],
    "C++": [
        ("nlohmann", "json", "develop", (".cpp", ".hpp", ".h")),
    ],
    "Ruby": [
        ("sinatra", "sinatra", "main", ".rb"),
    ]
}

# Pre-fetch file lists for each repo
repo_file_queues = {}
for lang, repos in LANGUAGE_REPOS.items():
    repo_file_queues[lang] = []
    for owner, repo, branch, ext in repos:
        tree_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
        try:
            req = urllib.request.Request(
                tree_url,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) CodeRoundRobinBot/1.0',
                    'Accept': 'application/vnd.github.v3+json'
                }
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                tree_data = json.loads(resp.read().decode('utf-8'))
                paths = [
                    (owner, repo, branch, item['path']) 
                    for item in tree_data.get('tree', [])
                    if item.get('type') == 'blob' and item.get('path', '').endswith(ext)
                ]
                repo_file_queues[lang].extend(paths)
                print(f"[{lang}] Loaded {len(paths)} files from {repo}", flush=True)
        except Exception as e:
            print(f"[-] Error fetching tree for {repo}: {e}", flush=True)

success_count = 0
languages = list(LANGUAGE_REPOS.keys())

# Round-robin loop across languages
while any(repo_file_queues[lang] for lang in languages):
    for lang in languages:
        if not repo_file_queues[lang]:
            continue
            
        # Pop the next file for this language
        owner, repo, branch, filepath = repo_file_queues[lang].pop(0)
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{filepath}"
        
        try:
            file_req = urllib.request.Request(
                raw_url,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(file_req, timeout=10) as file_resp:
                content = file_resp.read().decode('utf-8', errors='ignore')
                
                if len(content.strip()) > 50:
                    formatted_code = f"\n\n# --- LANG: {lang} | REPO: {owner}/{repo} | FILE: {filepath} ---\n\n" + content.strip() + "\n"
                    
                    with open(output_filename, "a", encoding="utf-8") as f:
                        f.write(formatted_code)
                        
                    success_count += 1
                    current_size = os.path.getsize(output_filename) if os.path.exists(output_filename) else 0
                    print(f"[{lang}] Saved {repo}/{filepath} | Total: {success_count} | Size: {current_size / (1024*1024):.2f} MB", flush=True)
                    
            time.sleep(0.2)
            
        except Exception as e:
            time.sleep(0.05)
            continue

print(f"\nRound-robin multi-language scraping complete! Total files saved: {success_count}. Saved to {output_filename}.", flush=True)
