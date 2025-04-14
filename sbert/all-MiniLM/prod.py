# app.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import torch
import uvicorn
import time
import os
import logging
import numpy as np
from typing import List, Optional, Union
import psutil
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("embedding-service")

# Initialize FastAPI app
app = FastAPI(
    title="Production Embedding Service",
    description="High-performance embedding service using all-MiniLM-L6-v2",
    version="1.0.0"
)

# Add CORS middleware for local subnet access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in local subnet
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Prometheus metrics
EMBEDDING_REQUESTS = Counter('embedding_requests_total', 'Total embedding requests')
EMBEDDING_ERRORS = Counter('embedding_errors_total', 'Total embedding errors')
EMBEDDING_PROCESSING_TIME = Histogram('embedding_processing_seconds', 'Time spent processing embeddings', 
                                     buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10])
BATCH_SIZE = Histogram('embedding_batch_size', 'Embedding batch sizes', 
                      buckets=[1, 2, 5, 10, 20, 50, 100])
SYSTEM_MEMORY_USAGE = Gauge('system_memory_usage_percent', 'System memory usage percentage')
SYSTEM_CPU_USAGE = Gauge('system_cpu_usage_percent', 'System CPU usage percentage')

# Model configuration
MODEL_NAME = os.environ.get("MODEL_NAME", "all-MiniLM-L6-v2")
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "64"))
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_DIMENSION = 384  # Specific to all-MiniLM-L6-v2

# Load model
logger.info(f"Loading model {MODEL_NAME} on {DEVICE}...")
try:
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    logger.info(f"Model loaded successfully with embedding dimension: {EMBEDDING_DIMENSION}")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

# Data models
class EmbeddingRequest(BaseModel):
    text: Union[str, List[str]] = Field(..., description="Text or list of texts to embed")
    normalize: bool = Field(True, description="Whether to L2-normalize the embeddings")

class EmbeddingResponse(BaseModel):
    embeddings: Union[List[float], List[List[float]]] = Field(..., description="Generated embeddings")
    dimensions: int = Field(..., description="Number of dimensions in the embedding")
    process_time_ms: float = Field(..., description="Processing time in milliseconds")
    model: str = Field(..., description="Model used for embeddings")
    
# Update system metrics periodically
def update_system_metrics():
    SYSTEM_MEMORY_USAGE.set(psutil.virtual_memory().percent)
    SYSTEM_CPU_USAGE.set(psutil.cpu_percent(interval=0.1))

# Endpoints
@app.post("/embed", response_model=EmbeddingResponse)
async def create_embedding(request: EmbeddingRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(update_system_metrics)
    
    try:
        start_time = time.time()
        EMBEDDING_REQUESTS.inc()
        
        # Handle both single texts and batches
        is_batch = isinstance(request.text, list)
        texts = request.text if is_batch else [request.text]
        
        # Log batch size
        batch_size = len(texts)
        BATCH_SIZE.observe(batch_size)
        logger.info(f"Processing batch of {batch_size} texts")
        
        # Check batch size limit
        if batch_size > MAX_BATCH_SIZE:
            raise HTTPException(status_code=400, detail=f"Batch size exceeds maximum allowed ({MAX_BATCH_SIZE})")
        
        # Generate embeddings
        with torch.no_grad():
            embeddings = model.encode(
                texts, 
                convert_to_numpy=True, 
                normalize_embeddings=request.normalize
            )
        
        # Convert to list for JSON serialization
        embeddings_list = embeddings.tolist()
        
        # Return single embedding or batch based on input
        result = embeddings_list if is_batch else embeddings_list[0]
        
        process_time = time.time() - start_time
        process_time_ms = round(process_time * 1000, 2)
        
        # Record processing time
        EMBEDDING_PROCESSING_TIME.observe(process_time)
        
        logger.info(f"Processed in {process_time_ms}ms")
        
        return {
            "embeddings": result,
            "dimensions": EMBEDDING_DIMENSION,
            "process_time_ms": process_time_ms,
            "model": MODEL_NAME
        }
    except Exception as e:
        EMBEDDING_ERRORS.inc()
        logger.error(f"Error generating embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check(background_tasks: BackgroundTasks):
    background_tasks.add_task(update_system_metrics)
    
    # Run a simple inference to check model health
    try:
        model.encode("Health check test")
        return {
            "status": "healthy",
            "model": MODEL_NAME,
            "device": DEVICE,
            "embedding_dimensions": EMBEDDING_DIMENSION,
            "system_info": {
                "memory_usage": f"{psutil.virtual_memory().percent}%",
                "cpu_usage": f"{psutil.cpu_percent(interval=0.1)}%"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/")
async def root():
    return {
        "service": "Embedding API",
        "model": MODEL_NAME,
        "endpoints": {
            "/embed": "POST - Generate embeddings",
            "/health": "GET - Service health check",
            "/metrics": "GET - Prometheus metrics"
        }
    }

# Add middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} completed in {process_time:.4f}s")
    return response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    workers = int(os.environ.get("WORKERS", "1"))
    logger.info(f"Starting server on port {port} with {workers} workers")
    uvicorn.run("app:app", host="0.0.0.0", port=port, workers=workers)