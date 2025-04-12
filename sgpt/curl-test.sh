# Get embeddings and save to file
curl -X POST "http://localhost:8000/embed" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a sample text for embedding generation."}' \
  -o response.json

# View the first few values
cat response.json | jq '.embedding[0:5]'
