# Jet-Nemotron API

Production-ready FastAPI for Jet-Nemotron text generation.

## Features

- **Load Once, Generate Many**: Model loaded once on startup, reused for all requests
- **FastAPI**: Modern async web framework with automatic API documentation
- **Health Checks**: Monitor model status and performance
- **Batch Processing**: Generate multiple texts in one request
- **Error Handling**: Comprehensive error handling and logging
- **Performance Metrics**: Track generation time and tokens per second

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_api.txt
```

### 2. Start the API Server

```bash
python3 api.py
```

The server will:
- Load the Jet-Nemotron-2B model (takes ~30-40 seconds)
- Perform warmup
- Start serving on `http://localhost:8000`

### 3. Test the API

```bash
# Open in browser for interactive docs
open http://localhost:8000/docs

# Or use the test client
python3 test_api.py
```

## API Endpoints

### Health Check
```
GET /health
```

Returns model status, device, and setup time.

### Text Generation
```
POST /generate
```

Request body:
```json
{
  "prompt": "Explain quantum computing:",
  "max_new_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "do_sample": true
}
```

Response:
```json
{
  "generated_text": "Quantum computing is...",
  "tokens_generated": 100,
  "generation_time": 15.2,
  "tokens_per_second": 6.6
}
```

### Batch Generation
```
POST /generate-batch
```

Process multiple prompts in one request (max 10).

### API Documentation
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Performance

Based on RTX 3060 Laptop GPU:
- **Setup time**: ~37 seconds (one-time)
- **Generation**: ~15-20 seconds for 100 tokens
- **Speed**: ~5-7 tokens/second
- **Memory**: ~4GB VRAM (bfloat16)

## Production Deployment

### Docker Deployment

```dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3 python3-pip
COPY requirements_api.txt .
RUN pip3 install -r requirements_api.txt

# Copy API code
COPY api.py .
EXPOSE 8000

# Start API server
CMD ["python3", "api.py", "--host", "0.0.0.0"]
```

### Environment Variables

- `MODEL_NAME`: Override default model (default: "jet-ai/Jet-Nemotron-2B")
- `HOST`: Server host (default: "0.0.0.0")
- `PORT`: Server port (default: 8000)

### Scaling Considerations

1. **Single GPU**: One API instance per GPU
2. **Load Balancing**: Use nginx or similar for multiple instances
3. **Monitoring**: Track `/health` endpoint for model status
4. **Memory**: Ensure sufficient GPU memory (~4GB for 2B model)

## Usage Examples

### Python Client

```python
import requests

# Generate text
response = requests.post("http://localhost:8000/generate", json={
    "prompt": "Write a poem about AI:",
    "max_new_tokens": 50
})

result = response.json()
print(result["generated_text"])
```

### cURL Client

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain machine learning:",
    "max_new_tokens": 100
  }'
```

### JavaScript Client

```javascript
const response = await fetch('http://localhost:8000/generate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    prompt: 'What is artificial intelligence?',
    max_new_tokens: 100
  })
});

const result = await response.json();
console.log(result.generated_text);
```

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `400`: Invalid request parameters
- `503`: Model not loaded (server starting up)
- `500`: Internal server error (generation failed)

## Monitoring

Use the health endpoint to monitor:
```bash
watch -n 5 'curl -s http://localhost:8000/health | jq .'
```

## Troubleshooting

### "Model not loaded" error
- Wait for server to finish loading (~30-40 seconds)
- Check server logs for loading errors
- Verify CUDA availability and GPU memory

### "CUDA out of memory"
- Reduce batch size
- Ensure no other GPU processes are running
- Consider using smaller model or system with more VRAM

### Slow generation
- This is expected for RTX 3060 (~5-7 tok/s)
- For faster performance, upgrade to RTX 4090 or data center GPU
- Reduce `max_new_tokens` for faster responses

## Architecture

The API follows the production pattern from `production_example.py`:

1. **Startup**: Load model once (~37s)
2. **Warmup**: Initialize CUDA kernels
3. **Serve**: Process requests efficiently
4. **Reuse**: No model reloading between requests

This is **50x+ faster** than loading the model for each request.

## Files

- `api.py`: Main FastAPI application
- `test_api.py`: Test client script
- `requirements_api.txt`: API dependencies
- `production_example.py`: Original production example (for reference)