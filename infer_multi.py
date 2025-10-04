import os

from transformers import AutoTokenizer

from utils.data import load_data_100, filter_cached, load_docs
from utils.llm import init_tokenizer_client
from utils.setup import parse_token_count

def get_pred_multi(data, args, out_file):
    context_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct-1M")
    llm_tokenizer, llm_client = init_tokenizer_client(args.llm, args.port)
    
    template = open('prompts/multi_round_context.txt', encoding='utf-8').read()
    final_template = open('prompts/final_round_context.txt', encoding='utf-8').read()
    
    docs = load_docs()
    if args.retriever == 'qwen3_0.6':
        from retrievers.qwen import QwenRetriever
        
        retriever = QwenRetriever(args.emb_port)
    elif args.retriever == 'bm25':
        from retrievers.bm25 import BM25Retriever
        
        retriever = BM25Retriever()

    if args.ppr:
        from retrievers.ppr import PPRRetriever
        retriever = PPRRetriever(
            base_retriever=retriever,
            seed_k=args.k,
            alpha=args.alpha
        )
    
    for item in tqdm(data):
        ordered_did_list = []
        response_list = []
        summary_list = []
        query_list = [item['question']]  # Track all queries used
        
        for round_idx in range(args.num_rounds - 1):
            ordered_score_did = retriever(query_list[-1])

            ordered_dids, prompt = get_haystack_prompt(
                item,
                query_list[-1],
                docs,
                ordered_score_did,
                template,
                args.context_size,
                context_tokenizer,
                all_summaries=summary_list if round_idx > 0 else None
            )
            ordered_did_list.append(ordered_dids)
            output = query_llm(prompt, args.llm, llm_tokenizer, llm_client,
                               max_new_tokens=args.max_new_tokens, args=args, truncate=True)
            response = output.strip()
            summary, next_query = parse_llm_response(response)
            
            response_list.append(response)
            summary_list.append(summary)
            query_list.append(next_query)
        
        ordered_score_did = retriever(query_list[-1])
        ordered_dids, final_prompt = get_haystack_prompt(
            item,
            query_list[-1],
            docs,
            ordered_score_did,
            final_template,
            args.context_size,
            context_tokenizer,
            all_summaries=summary_list,  # Pass all summaries for final round
            is_final_round=True
        )
        ordered_did_list.append(ordered_dids)
        
        final_output = query_llm(final_prompt, args.llm, llm_tokenizer, llm_client,
                                 max_new_tokens=args.max_new_tokens, args=args, truncate=True)
        final_response = final_output.strip()
        response_list.append(final_response)
        
        # Save results
        out_item = {
            'id': item['id'],
            'question': item['question'],
            'answer': item['answer'],
            'answer_aliases': item['answer_aliases'],
            'ordered_did_list': ordered_did_list,
            'response_list': response_list,
            'summary_list': summary_list,
            'query_list': query_list,
            'pred': extract_answer(final_response)
        }
        
        with open(out_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(out_item, ensure_ascii=False) + '\n')

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
    data = filter_cached(out_file, data_all)
    if len(data) == 0:
        return
    
    get_pred_multi(data, args, out_file)

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