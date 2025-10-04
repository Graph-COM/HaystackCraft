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
    parser.add_argument("--port", type=int, default=8000,
                        help="Port for the local API server (default: 8000)")
    parser.add_argument("--retriever", type=str, required=True,
                        choices=["bm25", "qwen3_0.6"])
    parser.add_argument("--emb_port", type=int, required=False,
                        help="Port for the embedding server")
    parser.add_argument("--ppr", action="store_true")
    parser.add_argument("--context_size", type=parse_token_count, required=True,
                        choices=[8_000, 16_000, 32_000, 64_000, 96_000, 128_000, 256_000, 512_000, 1_000_000],
                        help="Target token size for the constructed context (e.g., 8K, 16K, 32K, 64K, 128K, 256K, 512K, 1M). "
                             "Also accepts plain integers. Required if --check_max_tokens_no_distraction is not set.")
    parser.add_argument("--num_rounds", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--order", type=str, default='descending_order')
    args = parser.parse_args()

    if args.retriever != "bm25":
        assert args.emb_port is not None
    
    main(args)