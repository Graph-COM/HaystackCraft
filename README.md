# HaystackCraft

[[Link to Paper]](./paper.pdf)

![fig](theme_figure.png)

## Table of Contents

- [Environment Setup](#environment-setup)
- [Static NIAH with Heterogeneous Retrieval Strategies](#static-niah-with-heterogeneous-retrieval-strategies)

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

We use vLLM for LLM serving, e.g.,

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --api-key token-abc123 --gpu-memory-utilization 0.95 --trust-remote-code --port 8000
```

For LLM inference,

```bash
python infer_static.py --llm MODEL_TO_EVALUATE --port PORT_YOU_USE_ABOVE --retriever RETRIEVER_FOR_HAYSTACK_CONSTRUCTION --context_size TARGET_CONTEXT_SIZE --order HAYSTACK_ORDERING
```

Additionally specify `--ppr` for graph-based reranking with Personalized PageRank (PPR) in haystack construction.
