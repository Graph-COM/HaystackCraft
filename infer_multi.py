import json
import os

from tqdm import tqdm
from transformers import AutoTokenizer

from utils.data import load_data_100, filter_cached, load_docs
from utils.llm import init_tokenizer_client, query_llm, extract_answer
from utils.setup import parse_token_count

def get_haystack_prompt(
    item, 
    query, 
    docs, 
    ordered_score_did, 
    template, 
    context_size, 
    context_tokenizer,
    all_summaries,
    is_final_round=False
):
    doc2score = {did: score for (score, did) in ordered_score_did}
    
    num_doc_count = 0
    docs_added = dict()
    
    golden_doc_ids = set(item['p2d_id'].values())
    sep_token_count = len(context_tokenizer.encode("\n\n"))

    golden_doc_str_list = []
    for did in golden_doc_ids:
        num_doc_count += 1
        doc_did = docs[did]
        doc_str = f"Article Title: {doc_did['title']}\n{doc_did['text']}"
        docs_added[did] = doc_str
        golden_doc_str_list.append(f"\n {doc_str}")
    golden_context = '\n\n'.join(golden_doc_str_list)
    golden_prompt = template.replace('$DOC$', golden_context).replace('$Q$', query)
    num_tokens_total = len(context_tokenizer.encode(golden_prompt)) + sep_token_count
    
    if num_tokens_total < context_size:
        for _, did in ordered_score_did:
            if did in docs_added:
                continue
            
            num_doc_count += 1
            
            doc_did = docs[did]
            doc_str_did = f"Article Title: {doc_did['title']}\n{doc_did['text']}"
            doc_did_tokens = context_tokenizer.encode(doc_str_did)
            stop_cutoff = context_size - sep_token_count
            if len(doc_did_tokens) + num_tokens_total >= stop_cutoff:
                doc_str_did = context_tokenizer.decode(doc_did_tokens[:context_size - num_tokens_total])
                doc_did_tokens = context_tokenizer.encode(doc_str_did)
            
            num_tokens_total += len(doc_did_tokens)
            assert num_tokens_total <= context_size
            docs_added[did] = doc_str_did
            num_tokens_total += sep_token_count

            if num_tokens_total >= context_size:
                break

    ordered_dids = list(docs_added.keys())
    ordered_dids.sort(key=lambda x: doc2score.get(x, float('-inf')), reverse=True)
    haystack_context = '\n\n'.join(
        [docs_added[did] for did in ordered_dids]
    )
    
    if is_final_round:
        # Final round: format for answering
        if all_summaries:
            opening_text = "Read your previous analyses and the following articles, answer the question below."
            prev_summary_text = "Previous Analyses:\n"
            for i, summary in enumerate(all_summaries):
                prev_summary_text += f"Round {i+1}: {summary}\n"
            prev_summary_text += "\n"
        else:
            opening_text = "Read the following articles and answer the question below."
            prev_summary_text = ""
        prompt = template.replace('$DOC$', haystack_context).replace('$Q$', query).replace('$PREV_SUMMARY$', prev_summary_text).replace('$OPENING$', opening_text)
    else:
        # Intermediate rounds: format for summarizing and generating next query
        if all_summaries:
            opening_text = "Read your previous analyses and the following articles. Analyze the question below."
            prev_summary_text = "Previous Analyses:\n"
            for i, summary in enumerate(all_summaries):
                prev_summary_text += f"Round {i+1}: {summary}\n"
            prev_summary_text += "\n"
            instruction_text = "Based on your previous analyses and the potentially new articles provided, summarize your findings related to the question and refine the question."
        else:
            opening_text = "Read the following articles and analyze the question below."
            prev_summary_text = ""
            instruction_text = "Based on the articles provided, summarize your findings related to the question and refine the question."
        
        prompt = template.replace('$DOC$', haystack_context).replace('$Q$', query).replace('$PREV_SUMMARY$', prev_summary_text).replace('$INSTRUCTION$', instruction_text).replace('$OPENING$', opening_text)
    
    return ordered_dids, prompt

def parse_llm_response(response):
    """
    Parse LLM response to extract summary and next query.
    Expected format:
    Summary: (summary text)
    Refined Question: (query text)
    
    If parsing fails, treat the whole response as both summary and next question.
    """    
    # Try to find Summary and Next Query sections
    summary_start = response.find("Summary:")
    query_start = response.find("Refined Question:")
    
    if summary_start != -1 and query_start != -1:
        # Both sections found
        summary = response[summary_start + len("Summary:"):query_start].strip()
        next_query = response[query_start + len("Refined Question:"):].strip()
        return summary, next_query
    elif summary_start != -1:
        # Only summary found
        summary = response[summary_start + len("Summary:"):].strip()
        return summary, summary
    elif query_start != -1:
        # Only query found
        next_query = response[query_start + len("Refined Question:"):].strip()
        return next_query, next_query
    else:
        # No structured format found, use whole response for both
        return response, response

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