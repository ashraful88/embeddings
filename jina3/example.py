from transformers import AutoModel

# Initialize the model
model_name = "jinaai/jina-embeddings-v3@f1944de8402dcd5f2b03f822a4bc22a7f2de2eb9"  # Example revision hash
#model = SentenceTransformer(model_name)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

texts = [
    "Follow the white rabbit.",  # English
    "Sigue al conejo blanco.",  # Spanish
    "Suis le lapin blanc.",  # French
    "跟着白兔走。",  # Chinese
    "اتبع الأرنب الأبيض.",  # Arabic
    "Folge dem weißen Kaninchen.",  # German
]

# When calling the `encode` function, you can choose a `task` based on the use case:
# 'retrieval.query', 'retrieval.passage', 'separation', 'classification', 'text-matching'
# Alternatively, you can choose not to pass a `task`, and no specific LoRA adapter will be used.
embeddings = model.encode(texts, task="text-matching")

# Compute similarities
print(embeddings[0] @ embeddings[1].T)
