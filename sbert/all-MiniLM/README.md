
# Embedding Service with all-MiniLM-L6-v2

This is a high-performance embedding service using the all-MiniLM-L6-v2 model from Sentence Transformers.

## Features

- Fast embedding generation using PyTorch and Sentence Transformers
- Prometheus integration for observability
- Health checks and system metrics

## API Endpoints
POST /v1/embed


## Setup

Build the Docker image

```docker build -t minilm-embedding-service .```

Run the container

```docker run -d --name minilm-embeddings -p 8000:8000 minilm-embedding-service```


## How to use 

Test the health endpoint

curl http://localhost:8000/health

Get embeddings for a sample text

```
curl -X POST "http://localhost:8000/embed" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a sample text to test the all-MiniLM-L6-v2 model."}'
  ```


### benchmark

run in a venv

```
pip install requests matplotlib psutil

python benchmark.py
```


## Benchmarking

Testing Environment
Hardware: MacBook Pro M2, 16GB RAM
Model: SGPT-125M
Docker Configuration: Python 3.9, FastAPI
Testing Tool: Custom Python benchmarking script
Test Date: April 12, 2025


![Benchmark Results](benchmark_graphs.png)
