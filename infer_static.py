import json
import os

from tqdm import tqdm

from utils.data import load_data, filter_cached, download_if_not_exists

# Set TOKENIZERS_PARALLELISM to false to avoid warnings with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_token_count(value):
    """Parse token count from string format like '64K', '128K' etc."""
    if isinstance(value, int):
        return value
    
    value = str(value).upper().strip()
    
    # Handle direct integer input
    if value.isdigit():
        return int(value)
    
    # Parse with suffixes
    if value.endswith('K'):
        return int(float(value[:-1]) * 1000)
    else:
        raise ValueError(f"Invalid token count format: {value}. Use formats like '64K', '128K', '1M', or plain integers.")

def format_context(context):
    """Format the context list into a single string."""
    return '\n\n'.join(
        [f"Article {idx+1}\n: {doc_str}" for idx, doc_str in enumerate(context)]
    )

def load_context_file(args, data_all):
    qid2context = dict()

    if args.ppr:
        context_file = f'data/{args.retriever}_ppr/seed_{args.k}_alpha_{args.alpha}/{args.context_size}.jsonl'
    else:
        context_file = f'data/{args.retriever}/{args.context_size}.jsonl'
    download_if_not_exists(context_file)
    print(f'Loading context file: {context_file}')
    
    with open(context_file, 'r') as f:
        for line in tqdm(f, desc='Loading contexts'):
            record = json.loads(line)
            q_id = record['id']
            docs_added = record['docs_added']
            doc_order = record[args.order]
            ordered_docs = [
                docs_added[did] for did in doc_order
            ]
            qid2context[q_id] = format_context(ordered_docs)
    
    for item in tqdm(data_all, desc='Inserting contexts'):
        item['context'] = qid2context[item['id']]

def main(args):
    if args.ppr:
        if args.retriever == "bm25":
            args.k = 10
        elif args.retriever == "qwen3_0.6":
            args.k = 5
        elif args.retriever == "hybrid_bm25_qwen3_0.6":
            args.k = 5
        else:
            raise NotImplementedError
        
        if args.retriever == "bm25":
            args.alpha = 0.5
        elif args.retriever == "qwen3_0.6":
            args.alpha = 0.5
        elif args.retriever == "hybrid_bm25_qwen3_0.6":
            args.alpha = 0.15
        else:
            raise NotImplementedError
        
        save_dir = f"results/{args.retriever}_ppr/seed_{args.k}_alpha_{args.alpha}/{args.llm}/{args.context_size}/{args.order}"
    else:
        save_dir = f"results/{args.retriever}/{args.llm}/{args.context_size}/{args.order}"
    os.makedirs(save_dir, exist_ok=True)
    
    out_file = os.path.join(save_dir, 'pred.jsonl')
    
    data_all = load_data()
    data = filter_cached(out_file, data_all)
    if len(data) == 0:
        return
    
    load_context_file(args, data)

if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("--llm", type=str, required=True,
                        choices=["Llama-3.1-8B-Instruct", 
                                 "Llama-3.1-70B-Instruct", 
                                 "Qwen2.5-7B-Instruct-1M", 
                                 "Qwen2.5-14B-Instruct-1M",
                                 "Qwen3-8B",
                                 "Qwen3-14B",
                                 "Qwen3-32B",
                                 "gemma-3-12b-it",
                                 "gemma-3-27b-it",
                                 "gpt-4.1-mini-2025-04-14",
                                 "o4-mini-2025-04-16",
                                 "gemini-2.5-flash-lite"])
    parser.add_argument("--base_timeout", type=int, default=60,
                        help="Base timeout in seconds for API requests (default: 60, will scale with context length)")
    parser.add_argument("--port", type=int,
                        help="Port for the local API server (default: 8000)")
    parser.add_argument("--retriever", type=str, required=True,
                        choices=["bm25", "qwen3_0.6", "hybrid_bm25_qwen3_0.6"])
    parser.add_argument("--ppr", action="store_true")
    parser.add_argument("--context_size", type=parse_token_count, required=True,
                        choices=[8_000, 16_000, 32_000, 64_000, 96_000, 128_000],
                        help="Target token size for the constructed context (e.g., 8K, 16K, 32K, 64K, 128K). "
                             "Also accepts plain integers.")
    parser.add_argument("--order", type=str, required=True, choices=[
        'descending_order',
        'permutation_1',
        'permutation_2',
        'permutation_3'
    ], help="Use descending_order for retrieval-ranked order. The rest are three random permutations.")
    args = parser.parse_args()
    main(args)
    