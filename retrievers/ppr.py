import csv
import igraph as ig
import numpy as np

from tqdm import tqdm

from utils.data import load_docs, download_if_not_exists

def load_network_edges_direct(doc_to_idx):
    edge_file = 'data/edge_list.csv'
    download_if_not_exists(edge_file)
    
    edges = set()
    with open(edge_file, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ['source_node', 'target_node']
        
        desc = "Loading edges for graph construction and converting to vertex indices"
            
        for row in tqdm(reader, desc=desc):
            u = doc_to_idx[row[0]]
            v = doc_to_idx[row[1]]
            edges.add((u, v))
            edges.add((v, u))
    
    return list(edges)

def get_graph():
    docs = load_docs()
    
    all_doc_ids = list(docs.keys())
    
    # Construct the graph once outside the loop using igraph (much faster)
    print("Constructing graph...")

    # Create mapping from doc IDs to vertex indices
    doc_to_idx = {doc_id: i for i, doc_id in enumerate(all_doc_ids)}
    idx_to_doc = {i: doc_id for i, doc_id in enumerate(all_doc_ids)}

    # Load edges with direct conversion to vertex indices
    edges = load_network_edges_direct(doc_to_idx)

    # Create igraph directed graph
    G = ig.Graph(directed=True)
    G.add_vertices(len(all_doc_ids))
    G.add_edges(edges)  # Batch addition is much faster
    print(f"Constructed graph has {G.vcount()} nodes and {G.ecount()} edges")
    
    return G, doc_to_idx, idx_to_doc

def run_ppr(seed_docs_scores, doc_to_idx, graph, idx_to_doc, alpha, top_k=10000):  
    # Vectorized assignment of seed scores
    valid_seeds = []
    valid_scores = []
    
    for score, doc_id_str in seed_docs_scores:
        if doc_id_str in doc_to_idx and score > 0:
            valid_seeds.append(doc_to_idx[doc_id_str])
            valid_scores.append(score)
    assert len(valid_seeds) > 0

    # Prepare personalization vector using numpy for efficiency
    n_nodes = len(doc_to_idx)
    
    # Use numpy advanced indexing for efficient assignment
    valid_seeds = np.array(valid_seeds, dtype=np.int32)
    valid_scores = np.array(valid_scores, dtype=np.float32)
        
    personalization = np.zeros(n_nodes, dtype=np.float32)
    personalization[valid_seeds] = valid_scores
        
     # Vectorized normalization
    total_seed_score = np.sum(personalization)
    assert total_seed_score > 0
    personalization = personalization / total_seed_score
    
    # Convert to list for igraph compatibility
    personalization = personalization.tolist()
        
    ppr_scores_list = graph.personalized_pagerank(
        directed=True,
        damping=1.0 - alpha,
        reset=personalization,
    )
    
    # Get top-k indices with highest PPR scores using argpartition for efficiency
    ppr_scores_array = np.array(ppr_scores_list)
    
    # Use argpartition to get top-k elements efficiently (O(n) vs O(n log n))
    top_k_indices = np.argpartition(ppr_scores_array, -top_k)[-top_k:]
    # Sort only the top-k elements in descending order
    top_k_indices = top_k_indices[np.argsort(ppr_scores_array[top_k_indices])[::-1]]
    
    # Convert to ordered list format (in descending order of PPR scores)
    ppr_scores = [(ppr_scores_list[i], idx_to_doc[i]) for i in top_k_indices]
                
    return ppr_scores

class PPRRetriever:
    def __init__(self, base_retriever, seed_k, alpha):
        self.base_retriever = base_retriever
        self.seed_k = seed_k
        self.alpha = alpha
        
        self.G, self.doc_to_idx, self.idx_to_doc = get_graph()
    
    def __call__(self, query, k=2560):
        base_ordered_score_did = self.base_retriever(query, self.seed_k)
        ppr_ordered_score_did = run_ppr(
            base_ordered_score_did,
            self.doc_to_idx,
            self.G,
            self.idx_to_doc,
            self.alpha,
            k)
        
        return ppr_ordered_score_did
