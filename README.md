# HaystackCraft

[[Link to Paper]](./paper.pdf)

![fig](theme_figure.png)

## Table of Contents

- [Environment Setup](#environment-setup)

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