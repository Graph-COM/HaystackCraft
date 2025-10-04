import os

from utils.data import load_data_100
from utils.setup import parse_token_count

def main(args):
    if args.llm in ["gemini-2.5-flash-lite", "gemini-2.5-pro", "Qwen3-8B"]:
        args.max_new_tokens = None
    elif args.llm in ["Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct-1M", 
                      "gemma-3-12b-it"]:
        args.max_new_tokens = 512
    elif args.llm in ["gpt-5-mini-2025-08-07", "gpt-5-2025-08-07"]:
        args.max_new_tokens = 16382        
    else:
        raise NotImplementedError

    if args.ppr:
        if args.retriever == "bm25":
            args.k = 10
        elif args.retriever == "qwen3_0.6":
            args.k = 5
        else:
            raise NotImplementedError
        
        if args.retriever == "bm25":
            args.alpha = 0.5
        elif args.retriever == "qwen3_0.6":
            args.alpha = 0.5
        else:
            raise NotImplementedError
        
        save_dir = f"{args.num_rounds}_round_results/{args.retriever}_ppr/seed_{args.k}_alpha_{args.alpha}/{args.llm}/{args.context_size}/{args.order}"
    else:
        save_dir = f"{args.num_rounds}_round_results/{args.retriever}/{args.llm}/{args.context_size}/{args.order}"
    os.makedirs(save_dir, exist_ok=True)
    
    out_file = os.path.join(save_dir, 'pred.jsonl')
    
    data_all = load_data_100()

if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("--llm", type=str, required=True,
                        choices=["Llama-3.1-8B-Instruct", 
                                 "Qwen2.5-7B-Instruct-1M", 
                                 "gemma-3-12b-it",
                                 "Qwen3-8B",
                                 "gemini-2.5-flash-lite",
                                 "gemini-2.5-pro",
                                 "gpt-5-mini-2025-08-07",
                                 "gpt-5-2025-08-07"])
    parser.add_argument("--base_timeout", type=int, default=60,
                        help="Base timeout in seconds for API requests (default: 60, will scale with context length)")
    parser.add_argument("--port", type=int,
                        help="Port for the local API server")
    parser.add_argument("--retriever", type=str, required=True,
                        choices=["bm25", "qwen3_0.6"])
    parser.add_argument("--emb_port", type=int,
                        help="Port for the embedding server")
    parser.add_argument("--ppr", action="store_true")
    parser.add_argument("--context_size", type=parse_token_count, required=True,
                        choices=[8_000, 16_000, 32_000, 64_000, 128_000],
                        help="Target token size for the constructed context (e.g., 8K, 16K, 32K, 64K, 128K). "
                             "Also accepts plain integers.")
    parser.add_argument("--num_rounds", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--order", type=str, default='descending_order')
    args = parser.parse_args()

    if args.retriever != "bm25":
        assert args.emb_port is not None
    
    main(args)