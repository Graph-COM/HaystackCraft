import json
import re
import tiktoken
import time

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

maxlen_map = {
    "gpt-4.1-mini-2025-04-14": 1_000_000,
    "gpt-5-mini-2025-08-07": 400_000,
    "gpt-5-2025-08-07": 400_000,
    "o4-mini-2025-04-16": 200_000,
    "Llama-3.1-8B-Instruct": 128_000,
    "Llama-3.1-70B-Instruct": 128_000,
    "Qwen2.5-7B-Instruct-1M": 1_000_000,
    "Qwen2.5-14B-Instruct-1M": 1_000_000,
    "Qwen3-8B": 131_072,
    "Qwen3-14B": 131_072,
    "Qwen3-32B": 131_072,
    "gemma-3-12b-it": 128_000,
    "gemma-3-27b-it": 128_000,
    "gemini-2.5-pro": 1_048_576,
    "gemini-2.5-flash-lite": 1_048_576,
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

def query_llm(prompt, model, tokenizer, client, max_new_tokens, args, truncate=True):
    """
    Unified function to query language models with proper truncation and error handling.
    
    Args:
        prompt: The input prompt to send to the model
        model: Model name/identifier
        tokenizer: Tokenizer for encoding/decoding text
        client: API client for model access
        max_new_tokens: Maximum number of tokens to generate
        args: Additional arguments including max_len, temperature, base_timeout, etc.
    
    Returns:
        Generated text from the model or empty string if failed
    """
    max_len = maxlen_map[model]
    
    # Truncate input if needed
    if model in model_map:
        input_ids = tokenizer.encode(prompt)
        if len(input_ids) > max_len:
            if truncate:
                input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
                prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
            else:
                raise ValueError(f"Input length {len(input_ids)} exceeds max length {max_len} for model {model}.")
    else:
        input_ids = tokenizer.encode(prompt, disallowed_special=())
        if len(input_ids) > max_len:
            if truncate:
                # If input exceeds max length, keep first and last halves of the tokens.
                # This preserves both the beginning context and the recent/relevant ending
                input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
                prompt = tokenizer.decode(input_ids)
            else:
                raise ValueError(f"Input length {len(input_ids)} exceeds max length {max_len} for model {model}.")
        
    tries = 0
    if model in model_map:
        model = model_map[model]
    
    # Calculate dynamic timeout based on input length
    input_length = len(input_ids)
    base_timeout = getattr(args, 'base_timeout', 60) if args is not None else 60
    # Use a more reasonable scaling function that doesn't grow linearly for very large inputs
    # This caps the maximum timeout at a reasonable value while still scaling with input size
    max_additional_timeout = 1800  # Maximum additional timeout of 30 minutes
    scaling_factor = min(input_length / 1000, 1000) * 3.6  # Cap at 1M tokens for timeout calculation
    dynamic_timeout = base_timeout + min(scaling_factor, max_additional_timeout)

    while tries < 5:
        tries += 1
        try:
            if model in ["gpt-4.1-mini-2025-04-14", "o4-mini-2025-04-16", 
                         "gpt-5-mini-2025-08-07", "gpt-5-2025-08-07"]:
                additional_kwargs = {}
                if model in ["gpt-5-mini-2025-08-07", "gpt-5-2025-08-07"]:
                    additional_kwargs['reasoning_effort'] = 'medium'
                
                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=16382,
                    timeout=dynamic_timeout,  # Dynamic timeout based on context length
                    **additional_kwargs
                )
                return completion.choices[0].message.content
            elif model in ["gemini-2.5-pro", "gemini-2.5-flash-lite"]:
                from google.genai import types
                response = client.models.generate_content(
                    model=model, 
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=-1)
                    )
                )
                if response.text is None:
                    if response.prompt_feedback.block_reason[:18] == 'PROHIBITED_CONTENT':
                        return ''
                    else:
                        raise ValueError
                else:
                    return response.text
            else:
                if model.startswith("Qwen/Qwen3"):
                    additional_kwargs = {
                        'max_tokens': 32768,
                        'temperature': 0.6,
                        'top_p': 0.95,
                        'extra_body': {
                            "top_k": 20, 
                            "chat_template_kwargs": {"enable_thinking": True},
                        }
                    }
                else:
                    additional_kwargs = {
                        'max_tokens': max_new_tokens, 'timeout': dynamic_timeout
                    }
                
                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    **additional_kwargs
                )
            
                return completion.choices[0].message.content
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print("Error Occurs: \"%s\"        Retry ..."%(str(e)))
            time.sleep(20)
    else:
        print("Max tries. Failed.")
        return ''

def extract_answer(response):
    # Remove asterisks that might interfere with pattern matching
    response = response.replace('*', '').strip()
    flags = re.IGNORECASE # Add ignorecase flag

    # Pattern 1: Answer in parentheses: "The correct answer is[:] (Answer)"
    match = re.search(r'The correct answer is\s*:?\s+\(([^)]+)\)', response, flags)
    if match:
        return match.group(1).strip()

    # Pattern 2: Answer in double quotes: "The correct answer is[:] "Answer""
    match = re.search(r'The correct answer is\s*:?\s+"([^"]+)"', response, flags)
    if match:
        return match.group(1).strip()

    # Pattern 3: Answer follows directly, not starting with ( or "
    # Ends with period, newline, or end of string. Use non-greedy matching.
    match = re.search(r'The correct answer is\s*:?\s+([^(^"].*?)(?:\.|$|\n)', response, flags)
    if match:
        answer = match.group(1).strip()
        # Ensure something was actually captured
        if answer:
            return answer

    # Pattern 4: Handle truncated parenthesis case: "The correct answer is[:] (Answer"
    # This is checked after Pattern 1 fails.
    match = re.search(r'The correct answer is\s*:?\s+\((.+)', response, flags)
    if match:
        # Strip potential trailing punctuation that might belong outside the intended answer
        answer = match.group(1).strip().rstrip('.').strip()
        return answer

    # Pattern 5: Handle truncated quotes case: "The correct answer is[:] "Answer"
    # This is checked after Pattern 2 fails.
    match = re.search(r'The correct answer is\s*:?\s+"(.+)', response, flags)
    if match:
        # Strip potential trailing quote and punctuation
        answer = match.group(1).strip().rstrip('"').rstrip('.').strip()
        return answer

    # Fallback 1: General case - capture anything after "The correct answer is[:]" until end/newline/period
    match = re.search(r'The correct answer is\s*:?\s+(.+?)(\.|$|\n)', response, flags)
    if match:
        answer = match.group(1).strip()
        # Clean potential leading symbols only if they weren't part of a matched pair from patterns 1 or 2
        if answer.startswith('(') and not re.search(r'The correct answer is\s*:?\s+\(([^)]+)\)', response, flags):
             answer = answer.lstrip('(').strip()
        elif answer.startswith('"') and not re.search(r'The correct answer is\s*:?\s+"([^"]+)"', response, flags):
             answer = answer.lstrip('"').strip()
        # Ensure something was actually captured after potential stripping
        if answer:
            return answer

    # Fallback 2: Match everything after "The correct answer is[:]" to the end of the string
    match = re.search(r'The correct answer is\s*:?\s+(.+)', response, flags)
    if match:
        answer = match.group(1).strip()
        # Same cleaning logic as Fallback 1
        if answer.startswith('(') and not re.search(r'The correct answer is\s*:?\s+\(([^)]+)\)', response, flags):
             answer = answer.lstrip('(').strip()
        elif answer.startswith('"') and not re.search(r'The correct answer is\s*:?\s+"([^"]+)"', response, flags):
             answer = answer.lstrip('"').strip()
        # Ensure something was actually captured after potential stripping
        if answer:
            return answer

    return None # Return None if no pattern matches

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
