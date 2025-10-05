import faiss
import numpy as np
import os
import time
import torch

from openai import OpenAI
from tqdm import tqdm

from utils.data import download_dir_if_not_exists

def init_emb_client(port):
    client = OpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="token-abc123")
    return client

def index_corpus_emb():
    all_embeddings = {}
    
    # Progress bar for directories
    for i in tqdm(range(1, 8), desc="Loading directories"):
        path_dir = f'data/qwen3_0.6_docs_part_{i}'
        
        # Download directory if it doesn't exist
        download_dir_if_not_exists(path_dir)
        
        # Load all .npz files in the directory.
        for filename in tqdm(os.listdir(path_dir)):
            assert filename.endswith('.npz')
            file_path = os.path.join(path_dir, filename)
            data_file = np.load(file_path)
            
            ids_file = data_file['ids']
            embs_file = data_file['embs']
            
            # Progress bar for embeddings within each file
            for j in range(len(ids_file)):
                all_embeddings[str(int(ids_file[j]))] = embs_file[j]
    
    print(f"Loaded {len(all_embeddings)} embeddings")
    
    doc_ids = list(all_embeddings.keys())

    # Determine embedding dimensionality from the first vector
    emb_dim = all_embeddings[doc_ids[0]].shape[0]

    # Assemble the embedding matrix
    doc_mat = np.empty((len(doc_ids), emb_dim), dtype=np.float32)
    for idx, d_id in enumerate(tqdm(doc_ids, desc="Normalizing embeddings", unit="docs")):
        vec = all_embeddings[d_id]
        # L2-normalize for cosine similarity
        norm = np.linalg.norm(vec)
        if norm == 0:
            doc_mat[idx] = vec  # all-zeros, rare but handle
        else:
            doc_mat[idx] = vec / norm

    # Create a FAISS index for inner-product similarity (equivalent to cosine
    # after normalisation). We always start with a CPU index and optionally
    # move it to GPU.
    cpu_index = faiss.IndexFlatIP(emb_dim)
    cpu_index.add(doc_mat)
    
    return doc_ids, cpu_index

def get_detailed_instruct(query):
    task_description = 'Given a query, retrieve relevant documents that answer the query'
    return f'Instruct: {task_description}\nQuery:{query}'

def get_emb(
    client,
    text_list
):
    tries = 0
    while tries < 5:
        tries += 1
        try:                
            response = client.embeddings.create(
                model="Qwen/Qwen3-Embedding-0.6B",
                input=text_list,
                timeout=60,
            )
            batch_emb = [response.data[i].embedding 
                         for i in range(len(response.data))]
            
            return torch.tensor(batch_emb)
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print("Error Occurs: \"%s\"        Retry ..."%(str(e)))
            time.sleep(1)
    else:
        print("Max tries. Failed.")
        raise ValueError

class QwenRetriever:
    def __init__(self, emb_port):
        self.emb_client = init_emb_client(emb_port)
        self.doc_ids, self.corpus_index = index_corpus_emb()
    
    def __call__(self, query, k=2560):
        query = get_detailed_instruct(query)
        query_emb = get_emb(self.emb_client, [query]).numpy()
        query_norm = np.linalg.norm(query_emb)
        if query_norm != 0:
            query_emb = query_emb / query_norm
            
        D, I = self.corpus_index.search(query_emb.reshape(1, -1), k)

        ordered_score_did = []
        for score, idx_retrieved in zip(D[0].tolist(), I[0].tolist()):
            doc_id = self.doc_ids[idx_retrieved]
            ordered_score_did.append((float(score), doc_id))
        
        return ordered_score_did
