import json
import os

from huggingface_hub import hf_hub_download

def download_if_not_exists(filename, repo_id="Graph-COM/HaystackCraft", local_dir="."):
    local_path = filename
    if not os.path.exists(local_path):
        print(f"{local_path} not found locally. Downloading from {repo_id}...")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=local_dir,
                local_dir_use_symlinks=False # Recommended for compatibility
            )
            print(f"Successfully downloaded {filename} to {local_path}")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
    else:
        print(f"Found {local_path} locally.")
    return local_path

def load_data():
    data = []

    for file in ['musique_400_250616.jsonl',
                 'nq_filtered.jsonl']:
        full_file = os.path.join('data', file)
        download_if_not_exists(full_file)
        
        with open(full_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    
    return data
