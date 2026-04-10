# Ray Inference Examples

A collection of examples demonstrating how to serve LLMs and embedding models using **Ray Serve**, **vLLM**, and **FastAPI**.

## Overview

This repository showcases different patterns for model inference at scale:

| Example | File | Description |
|---------|------|-------------|
| Multi-model serving | `llm_app.py` | Serve multiple LLMs with independent autoscaling and GPU allocation using Ray Serve's built-in `LLMConfig` |
| Custom FastAPI ingress | `llm_app_ingress.py` | Route requests to different model deployments via a FastAPI gateway with OpenAI-compatible streaming |
| Streaming with vLLM | `llm_streaming.py` | Direct vLLM `AsyncLLMEngine` integration with token-by-token streaming |
| OpenAI-compatible serving | `openai_serving.py` | Full OpenAI protocol compliance using vLLM's native `OpenAIServingChat` layer |
| vLLM OpenAI standalone | `openai_serving_example.py` | Standalone vLLM OpenAI serving without Ray Serve |
| Embedding with batching | `dynamic_batching.py` | Embedding model with Ray Serve's `@serve.batch()` for automatic dynamic batching |

## Models Used

- **LLM**: `HuggingFaceTB/SmolLM2-135M-Instruct` (135M params, lightweight for experimentation)
- **Embeddings**: `Qwen/Qwen3-Embedding-0.6B`

## Getting Started

### Prerequisites

- Python 3.11+
- CUDA-capable GPU
- Key dependencies: `ray[serve]`, `vllm`, `transformers`, `torch`, `fastapi`, `openai`, `httpx`

### Running an example

```bash
# Multi-model LLM serving
serve run llm_app:llm_app

# Custom ingress with mock LLMs
serve run llm_app_ingress:llm_app

# Streaming inference
serve run llm_streaming:app

# OpenAI-compatible serving
serve run openai_serving:app

# Embedding service with dynamic batching
serve run dynamic_batching:app
```

### Testing

```bash
# Simple chat completion request
python client.py

# OpenAI SDK client
python openai_client.py

# Streaming response test
python test_streaming_server.py

# Concurrent embedding requests (512 async)
python embedding_client.py

# Direct vLLM engine test
python test.py
```

## Kubernetes Deployment

The `ray-tiny.yaml` file provides a `RayService` spec for deploying on Kubernetes with:

- Ray 2.46.0
- GPU worker nodes (`rtx2050-worker-group`)
- Local Docker image built from the included `dockerfile`

```bash
# Build the image
docker build -t ray-llm-local:latest -f dockerfile .

# Deploy
kubectl apply -f ray-tiny.yaml
```

## Project Structure

```
.
├── llm_app.py                 # Multi-model LLM serving (Ray Serve LLMConfig)
├── llm_app_ingress.py         # FastAPI ingress + mock LLM deployments
├── llm_streaming.py           # vLLM streaming inference
├── openai_serving.py          # OpenAI-compatible serving via vLLM
├── openai_serving_example.py  # Standalone vLLM OpenAI example
├── dynamic_batching.py        # Embedding model with dynamic batching
├── client.py                  # HTTP test client
├── openai_client.py           # OpenAI SDK test client
├── embedding_client.py        # Async concurrent embedding test client
├── test_streaming_server.py   # Streaming response test client
├── test.py                    # Direct vLLM engine test
├── ray-tiny.yaml              # Kubernetes RayService manifest
└── dockerfile                 # Container image definition
```
