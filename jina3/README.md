# Build the Docker image
docker build -t jina-v3-embedding-service .

# Run the container
docker run -d --name jina-v3-embeddings -p 8000:8000 jina-v3-embedding-service

# Test the health endpoint
curl http://localhost:8000/health

# Get embeddings for a sample text
curl -X POST "http://localhost:8000/embed" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a sample text to test the Jina embeddings v3 model."}'

## Benchmarking

Testing Environment
Hardware: MacBook Pro M2, 16GB RAM
Model: SGPT-125M
Docker Configuration: Python 3.9, FastAPI
Testing Tool: Custom Python benchmarking script
Test Date: April 12, 2025


![Benchmark Results](benchmark_graphs.png)
