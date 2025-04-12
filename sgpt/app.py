# Example FastAPI implementation
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer
import torch
import uvicorn

app = FastAPI(title="SGPT Embedding Service")

# Load model (could be SGPT-125M, SGPT-570M, etc.)
model_name = "Muennighoff/SGPT-125M-weightedmean-nli-bitfit"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

class EmbeddingRequest(BaseModel):
    text: str
    
@app.post("/embed")
async def get_embedding(request: EmbeddingRequest):
    try:
        inputs = tokenizer(request.text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        # Use mean pooling to get embeddings
        embeddings = outputs.last_hidden_state.mean(dim=1).tolist()[0]
        return {"embedding": embeddings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)