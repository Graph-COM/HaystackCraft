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
    