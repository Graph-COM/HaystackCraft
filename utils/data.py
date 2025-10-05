import json
import os

from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm

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

def load_data_100():
    data = []
    
    full_file = os.path.join('data', '100_for_multi.jsonl')
    download_if_not_exists(full_file)
    
    with open(full_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    return data

def filter_cached(out_file, data_all):
    ids_done = set()
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            ids_done = set([
                json.loads(line)["id"] for line in f
            ])
    
    data = []
    for item in data_all:
        if item["id"] not in ids_done:
            data.append(item)
    
    return data

def load_docs():
    doc_file = 'data/docs.jsonl'
    download_if_not_exists(doc_file)
    
    docs = {}
    with open(doc_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading documents"):
            doc = json.loads(line)
            doc_id = doc['id']
            assert doc_id not in docs
            docs[doc_id] = doc
    
    return docs

def download_dir_if_not_exists(dirname, repo_id="Graph-COM/HaystackCraft", local_dir="."):
    local_path = dirname
    if not os.path.exists(local_path):
        print(f"{local_path} not found locally. Downloading from {repo_id}...")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                allow_patterns=[f"{dirname}/*"]
            )
            print(f"Successfully downloaded {dirname} to {local_path}")
        except Exception as e:
            print(f"Failed to download {dirname}: {e}")
    else:
        print(f"Found {local_path} locally.")
    return local_path
