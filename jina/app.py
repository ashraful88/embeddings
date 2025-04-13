# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch
import uvicorn

app = FastAPI(title="Jina AI Embedding Service")

# Load the Jina embedding model
model_name = "jinaai/jina-embeddings-v2-small-en"
model = SentenceTransformer(model_name)

class EmbeddingRequest(BaseModel):
    text: str
    
@app.post("/embed")
async def get_embedding(request: EmbeddingRequest):
    try:
        # Generate embedding
        embedding = model.encode(request.text, convert_to_tensor=True)
        # Convert to list for JSON serialization
        embedding_list = embedding.tolist() if isinstance(embedding, torch.Tensor) else embedding.tolist()[0]
        return {"embedding": embedding_list, "dimensions": len(embedding_list)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)