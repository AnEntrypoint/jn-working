"""
FastAPI for Jet-Nemotron text generation
Load model once, serve many requests efficiently
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import time
import logging
import asyncio
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None
MODEL_NAME = "jet-ai/Jet-Nemotron-2B"

class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

class GenerationResponse(BaseModel):
    generated_text: str
    tokens_generated: int
    generation_time: float
    tokens_per_second: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    setup_time: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup and cleanup on shutdown"""
    global model, tokenizer

    logger.info("Starting model loading...")
    setup_start = time.time()

    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="flash_attention_2"
        )
        model.eval()

        # Warmup
        logger.info("Performing warmup...")
        warmup_input = tokenizer("Hello", return_tensors="pt").to("cuda")
        with torch.no_grad():
            _ = model.generate(**warmup_input, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)

        setup_time = time.time() - setup_start
        logger.info(f"Model loaded and warmed up in {setup_time:.2f}s")

        # Store setup time for health checks
        app.state.setup_time = setup_time

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    yield

    logger.info("Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="Jet-Nemotron API",
    description="Production-ready API for Jet-Nemotron text generation",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API is healthy and model is loaded"""
    global model, tokenizer

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        device=str(model.device),
        setup_time=app.state.setup_time
    )

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """Generate text using Jet-Nemotron model"""
    global model, tokenizer

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Tokenize input
        inputs = tokenizer(request.prompt, return_tensors="pt").to("cuda")

        # Generate
        torch.cuda.synchronize()
        gen_start = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=request.do_sample,
                temperature=request.temperature,
                top_p=request.top_p,
                pad_token_id=tokenizer.eos_token_id
            )

        torch.cuda.synchronize()
        gen_time = time.time() - gen_start

        # Calculate metrics
        tokens_generated = outputs.shape[1] - inputs['input_ids'].shape[1]
        tokens_per_second = tokens_generated / gen_time if gen_time > 0 else 0

        # Decode response
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return GenerationResponse(
            generated_text=result,
            tokens_generated=tokens_generated,
            generation_time=gen_time,
            tokens_per_second=tokens_per_second
        )

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/generate-batch", response_model=List[GenerationResponse])
async def generate_batch(requests: List[GenerationRequest]):
    """Generate text for multiple prompts"""
    global model, tokenizer

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(requests) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Batch size limited to 10 requests")

    results = []
    for request in requests:
        try:
            # Tokenize input
            inputs = tokenizer(request.prompt, return_tensors="pt").to("cuda")

            # Generate
            torch.cuda.synchronize()
            gen_start = time.time()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=request.max_new_tokens,
                    do_sample=request.do_sample,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    pad_token_id=tokenizer.eos_token_id
                )

            torch.cuda.synchronize()
            gen_time = time.time() - gen_start

            # Calculate metrics
            tokens_generated = outputs.shape[1] - inputs['input_ids'].shape[1]
            tokens_per_second = tokens_generated / gen_time if gen_time > 0 else 0

            # Decode response
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)

            results.append(GenerationResponse(
                generated_text=result,
                tokens_generated=tokens_generated,
                generation_time=gen_time,
                tokens_per_second=tokens_per_second
            ))

        except Exception as e:
            logger.error(f"Batch generation failed for prompt '{request.prompt[:50]}...': {e}")
            # Add a failed response
            results.append(GenerationResponse(
                generated_text=f"Error: {str(e)}",
                tokens_generated=0,
                generation_time=0,
                tokens_per_second=0
            ))

    return results

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "Jet-Nemotron API",
        "model": MODEL_NAME,
        "endpoints": {
            "health": "/health",
            "generate": "/generate",
            "generate_batch": "/generate-batch",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)