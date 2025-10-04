import tiktoken

from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

model_map = {
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "Llama-3.1-70B-Instruct": "meta-llama/Llama-3.1-70B-Instruct",
    "Qwen2.5-7B-Instruct-1M": "Qwen/Qwen2.5-7B-Instruct-1M",
    "Qwen2.5-14B-Instruct-1M": "Qwen/Qwen2.5-14B-Instruct-1M",
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Qwen3-14B": "Qwen/Qwen3-14B",
    "Qwen3-32B": "Qwen/Qwen3-32B",
    "gemma-3-12b-it": "google/gemma-3-12b-it",
    "gemma-3-27b-it": "google/gemma-3-27b-it",
}

def init_tokenizer_client(model, port):
    if model in ["gpt-4.1-mini-2025-04-14", "o4-mini-2025-04-16",
                 "gpt-5-mini-2025-08-07", "gpt-5-2025-08-07"]:
        try:
            tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            # For newer models not in tiktoken's automatic mapping, use cl100k_base encoding
            tokenizer = tiktoken.get_encoding("cl100k_base")
        client = OpenAI()
    elif model in ["gemini-2.5-pro", "gemini-2.5-flash-lite"]:
        from google import genai
        client = genai.Client()
        tokenizer = tiktoken.get_encoding("cl100k_base")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_map[model], trust_remote_code=True)
        client = OpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key="token-abc123",
        )
    return tokenizer, client

def get_pred(data, args, out_file):
    tokenizer, client = init_tokenizer_client(args.llm, args.port)
    template = open('prompts/context.txt', encoding='utf-8').read()
    
    with open(out_file, 'a', encoding='utf-8') as fout:
        for item in tqdm(data):
            prompt = template.replace('$DOC$', item['context']).replace('$Q$', item['question'])

            output = query_llm(prompt, args.llm, tokenizer, client, 
                               max_new_tokens=512, args=args, truncate=True)
            response = output.strip()
        
            out_item = {
                'id': item['id'],
                'question': item['question'],
                'answer': item['answer'],
                'answer_aliases': item['answer_aliases'],
                'response': response,
                'pred': extract_answer(response)
            }

            fout.write(json.dumps(out_item, ensure_ascii=False) + '\n')
            fout.flush()
