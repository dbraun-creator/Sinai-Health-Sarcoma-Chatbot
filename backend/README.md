# Sarcoma Q&A Backend - Semantic Search API

## Overview
This backend provides a semantic search API for sarcoma-related questions and answers. It uses OpenAI's embedding model to convert user queries into vectors and performs cosine similarity search against a pre-computed dataset of medical Q&A pairs.

## Architecture

### Modular Design
- **`app.py`**: Main Flask application with API endpoints
- **`data_loader.py`**: Abstract data loading with CSV implementation (easily extensible to databases)
- **`embedding_service.py`**: OpenAI embedding generation with caching support
- **`search_service.py`**: Cosine similarity search logic
- **`config.py`**: Centralized configuration management

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file in the backend directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
ENVIRONMENT=development  # or production
```

### 3. Run Locally
```bash
python app.py
```
The server will start on `http://localhost:5000`

## API Endpoints

### 1. Health Check
```http
GET /health
```
Returns server status and dataset size.

### 2. Search (Main Endpoint)
```http
POST /search
Content-Type: application/json

{
    "query": "What are the symptoms of sarcoma?",
    "threshold": 0.7  // Optional, default: 0.7
}
```

**Response:**
```json
{
    "answer": "The answer text...",
    "source": "Source reference...",
    "similarity_score": 0.85,
    "matched_question": "Original question from dataset",
    "threshold_used": 0.7,
    "processing_time": 0.234
}
```

### 3. Batch Search
```http
POST /search/batch
Content-Type: application/json

{
    "queries": [
        "What is sarcoma?",
        "Treatment options?"
    ],
    "threshold": 0.7
}
```

### 4. Statistics
```http
GET /stats
```
Returns dataset and configuration statistics.

## Deployment on Render

### 1. Create `render.yaml`
```yaml
services:
  - type: web
    name: sarcoma-qa-backend
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "gunicorn app:app"
    envVars:
      - key: OPENAI_API_KEY
        sync: false
      - key: ENVIRONMENT
        value: production
```

### 2. Deploy
1. Push code to GitHub
2. Connect GitHub repo to Render
3. Add environment variables in Render dashboard
4. Deploy

## Performance Considerations

### Current Design (CSV-based)
- **Dataset**: ~140 rows with pre-computed embeddings
- **Memory Usage**: ~3-5 MB for embeddings in memory
- **Latency**: 
  - OpenAI API: 100-500ms
  - Cosine similarity: <10ms for 140 vectors
  - Total: 150-600ms per request

### Scalability Path
1. **Short-term (< 1000 entries)**: Current design is sufficient
2. **Medium-term (1000-10,000 entries)**: Add Redis caching
3. **Long-term (> 10,000 entries)**: Migrate to vector database
   - PostgreSQL with pgvector
   - Pinecone / Weaviate / Qdrant

## Testing

### Run Test Script
```python
# test_backend.py
import requests
import json

# Test search endpoint
url = "http://localhost:5000/search"
data = {
    "query": "What are the early symptoms of sarcoma?",
    "threshold": 0.7
}

response = requests.post(url, json=data)
print(json.dumps(response.json(), indent=2))
```

### Test Different Thresholds
```python
# Test threshold sensitivity
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
query = "sarcoma treatment options"

for threshold in thresholds:
    response = requests.post(url, json={
        "query": query,
        "threshold": threshold
    })
    result = response.json()
    print(f"Threshold {threshold}: Score={result.get('similarity_score', 0):.3f}")
```

## Future Enhancements

### 1. Database Migration
The `DataLoader` abstract class makes it easy to switch from CSV to database:
```python
# Future implementation
class PostgreSQLDataLoader(DataLoader):
    def __init__(self, connection_string):
        # Connect to PostgreSQL with pgvector extension
        pass
```

### 2. Caching Layer
```python
# Add Redis caching
from flask_caching import Cache
cache = Cache(app, config={'CACHE_TYPE': 'redis'})
```

### 3. Advanced Features
- Query expansion using synonyms
- Multi-language support
- Feedback loop for improving matches
- A/B testing different thresholds
- Real-time embedding updates

## Monitoring

### Recommended Metrics
- Response time per endpoint
- Similarity score distribution
- Cache hit rate
- API error rate
- Threshold effectiveness

### Error Handling
The backend includes comprehensive error handling:
- Invalid queries return 400
- API failures return appropriate error messages
- Fallback responses for low similarity scores

## Security Considerations
- API key stored in environment variables
- CORS configured for frontend domain
- Input validation on all endpoints
- Rate limiting (can be added via Flask-Limiter)

## Questions & Answers

**Q: Why not use a vector database immediately?**
A: With ~140 embeddings, the overhead of a vector database isn't justified. The current in-memory approach is fast and simple.

**Q: How to handle embedding dimension changes?**
A: The `EmbeddingService` automatically handles different dimensions. Switching models requires re-computing all embeddings.

**Q: Can this scale to thousands of questions?**
A: Yes, with minimal changes. Add Redis caching first, then consider a vector database beyond 10k entries.

## Support
For issues or questions, please refer to the main project repository.