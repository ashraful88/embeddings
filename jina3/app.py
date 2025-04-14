# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
#from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer, __version__ as sentence_transformers_version
import torch
import uvicorn
import time

app = FastAPI(title="Jina Embeddings v3 Service")

# Load the Jina v3 embedding model
print("Loading model...")
model_name = "jinaai/jina-embeddings-v3"
#model = SentenceTransformer(model_name)

print(sentence_transformers_version)
model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)
print(f"Model loaded: {model_name}")

class EmbeddingRequest(BaseModel):
    text: str
    
@app.post("/embed")
async def get_embedding(request: EmbeddingRequest):
    try:
        start_time = time.time()
        # Generate embedding
        embedding = model.encode(request.text)
        process_time = time.time() - start_time
        
        # Convert to list for JSON serialization
        embedding_list = embedding.tolist() if isinstance(embedding, torch.Tensor) else embedding.tolist()
        
        return {
            "embedding": embedding_list, 
            "dimensions": len(embedding_list),
            "process_time_ms": round(process_time * 1000, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "jinaai/jina-embeddings-v3"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)