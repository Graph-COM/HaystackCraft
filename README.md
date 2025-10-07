# Haystack Engineering: Context Engineering for Heterogeneous and Agentic Long-Context Evaluation

[[Link to Paper]](./paper.pdf)

![fig](theme_figure.png)

## Table of Contents

- [Environment Setup](#environment-setup)
- [Static NIAH with Heterogeneous Retrieval Strategies](#static-niah-with-heterogeneous-retrieval-strategies)
- [Dynamic NIAH](#dynamic-niah)
    * [Retrieval Environment Setup](#retrieval-environment-setup)
        + [BM25](#bm25)
        + [qwen3_0.6](#qwen3_06)
    * [LLM Inference (Enforced Multi-Round)](#llm-inference-enforced-multi-round)
    * [LLM Inference (Variable-Round)](#llm-inference-variable-round)
    * [Evaluation](#evaluation)

## Environment Setup

```bash
conda create -n HaystackCraft python=3.10 -y
conda activate HaystackCraft
pip install -r requirements.txt
```

If you have trouble running Qwen2.5-1M models, you may create a separate environment with `requirements_0-7-2.txt`.

If you need to evaluate models from OpenAI, specify your OpenAI API key with

```bash
export OPENAI_API_KEY=...
```

If you need to evaluate Gemini models, specify your Gemini API key with

```bash
export GEMINI_API_KEY=...
```

## Static NIAH with Heterogeneous Retrieval Strategies

For access to certain open source LLMs, you may need to first specify your huggingface token with `export HUGGING_FACE_HUB_TOKEN=...`.

We use vLLM for serving open source LLMs, e.g.,

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --api-key token-abc123 --gpu-memory-utilization 0.95 --trust-remote-code --port 8000
```

For LLM inference,

```bash
python infer_static.py --llm MODEL_TO_EVALUATE --retriever RETRIEVER_FOR_HAYSTACK_CONSTRUCTION --context_size TARGET_CONTEXT_SIZE --order HAYSTACK_ORDERING
```

Additionally specify `--ppr` for graph-based reranking with Personalized PageRank (PPR) in haystack construction.

For inference with locally deployed open source LLMs, specify the port you use in vLLM deployment, e.g., `--port 8000`.

For evaluation, do for example

```bash
python eval.py --result_dir results/bm25/Llama-3.1-8B-Instruct/8000/descending_order/
```

## Dynamic NIAH

### Retrieval Environment Setup

#### BM25

Install Java 21 with for example

```bash
curl -s "https://get.sdkman.io" | bash
source "$HOME/.sdkman/bin/sdkman-init.sh"
sdk install java 21.0.3-tem
```

#### qwen3_0.6

Deploy a local embedding server with vLLM.

```bash
vllm serve Qwen/Qwen3-Embedding-0.6B --port QWEN_RETRIEVER_EMB_PORT --api-key token-abc123 --gpu-memory-utilization 0.95 --trust-remote-code --enforce-eager
```

### LLM Inference (Enforced Multi-Round)

```bash
python infer_multi.py --llm MODEL_TO_EVALUATE --retriever RETRIEVER_FOR_HAYSTACK_CONSTRUCTION --context_size TARGET_CONTEXT_SIZE --num_rounds NUM_REASONING_ROUNDS
```

Additional args:
- `--port`: For inference with locally deployed open source LLMs, specify the port you use in vLLM deployment, e.g., `--port 8000`.
- `--emb_port`: If you use `Qwen3-Embedding-0.6B` for haystack construction, specify `QWEN_RETRIEVER_EMB_PORT` used above.
- `--ppr`: Specify `--ppr` for graph-based reranking with Personalized PageRank (PPR) in haystack construction.

### LLM Inference (Variable-Round)

```bash
python infer_variable.py --llm MODEL_TO_EVALUATE --retriever RETRIEVER_FOR_HAYSTACK_CONSTRUCTION --context_size TARGET_CONTEXT_SIZE --max_rounds MAX_REASONING_ROUNDS
```

Additional args:
- `--port`: For inference with locally deployed open source LLMs, specify the port you use in vLLM deployment, e.g., `--port 8000`.
- `--emb_port`: If you use `Qwen3-Embedding-0.6B` for haystack construction, specify `QWEN_RETRIEVER_EMB_PORT` used above.
- `--ppr`: Specify `--ppr` for graph-based reranking with Personalized PageRank (PPR) in haystack construction.

### Evaluation

For example

```bash
python eval_100.py --result_dir 2_round_results/qwen3_0.6/gemini-2.5-flash-lite/8000/descending_order
```
