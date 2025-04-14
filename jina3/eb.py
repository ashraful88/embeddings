import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


sentences = ["How is the weather today?", "What is the current weather like today?"]

tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3")
model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)

encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
task = 'retrieval.query'
task_id = model._adaptation_map[task]
adapter_mask = torch.full((len(sentences),), task_id, dtype=torch.int32)
with torch.no_grad():
    model_output = model(**encoded_input, adapter_mask=adapter_mask)

embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
embeddings = F.normalize(embeddings, p=2, dim=1)

print(embeddings)