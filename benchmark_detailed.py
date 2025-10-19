import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

print("=" * 80)
print("Detailed Performance Benchmark")
print("=" * 80)

MODEL_NAME = "jet-ai/Jet-Nemotron-2B"

# 1. Model Loading
print("\n1. MODEL LOADING")
print("-" * 80)
start = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer_time = time.time() - start
print(f"   Tokenizer load: {tokenizer_time:.2f}s")

start = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="flash_attention_2"
)
model_load_time = time.time() - start
print(f"   Model load:     {model_load_time:.2f}s")
print(f"   TOTAL LOADING:  {tokenizer_time + model_load_time:.2f}s")

# 2. Warmup (first run is always slower due to CUDA initialization)
print("\n2. WARMUP RUN")
print("-" * 80)
test_input = "Hello"
inputs = tokenizer(test_input, return_tensors="pt").to("cuda")

start = time.time()
with torch.no_grad():
    _ = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
warmup_time = time.time() - start
print(f"   Warmup generation: {warmup_time:.2f}s (includes CUDA init overhead)")

torch.cuda.empty_cache()

# 3. Detailed generation benchmark
print("\n3. ACTUAL INFERENCE BENCHMARK")
print("-" * 80)

test_prompt = "Write a short story about artificial intelligence:"
inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
prompt_tokens = inputs['input_ids'].shape[1]

print(f"   Prompt: '{test_prompt}'")
print(f"   Prompt tokens: {prompt_tokens}")

# Measure just the generation time (no loading)
torch.cuda.synchronize()
gen_start = time.time()

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

torch.cuda.synchronize()
gen_total = time.time() - gen_start

generated_tokens = outputs.shape[1] - prompt_tokens
total_tokens = outputs.shape[1]

print(f"\n   Generated tokens: {generated_tokens}")
print(f"   Total tokens:     {total_tokens}")
print(f"   Generation time:  {gen_total:.2f}s")
print(f"   Tokens/second:    {generated_tokens/gen_total:.1f} tok/s")

# Estimate prefill vs decode
# Rough estimate: prefill is processing all prompt tokens, decode is one-by-one
# Prefill happens once at start, decode happens for each new token
prefill_estimate = 0.5  # seconds (rough estimate)
decode_estimate = gen_total - prefill_estimate
decode_tokens_per_sec = generated_tokens / decode_estimate if decode_estimate > 0 else 0

print(f"\n   Estimated breakdown:")
print(f"   - Prefill (~first pass):  ~{prefill_estimate:.2f}s")
print(f"   - Decode ({generated_tokens} tokens): ~{decode_estimate:.2f}s")
print(f"   - Decode speed:           ~{decode_tokens_per_sec:.1f} tok/s")

# 4. Memory usage
print("\n4. MEMORY USAGE")
print("-" * 80)
mem_allocated = torch.cuda.max_memory_allocated() / 1024**3
mem_reserved = torch.cuda.max_memory_reserved() / 1024**3
print(f"   Peak allocated: {mem_allocated:.2f} GB")
print(f"   Peak reserved:  {mem_reserved:.2f} GB")

# 5. Summary
print("\n" + "=" * 80)
print("PERFORMANCE SUMMARY")
print("=" * 80)
print(f"Startup time (one-time):  {tokenizer_time + model_load_time:.2f}s")
print(f"Warmup time (first run):  {warmup_time:.2f}s")
print(f"Pure inference (200 tok): {gen_total:.2f}s ({generated_tokens/gen_total:.1f} tok/s)")
print(f"\nFor production use:")
print(f"- Load model once:        {model_load_time:.2f}s (one-time)")
print(f"- Then each generation:   ~{gen_total:.2f}s for {generated_tokens} tokens")
print(f"- Effective throughput:   {generated_tokens/gen_total:.1f} tokens/second")
print("=" * 80)

# 6. Generated text
print("\n5. GENERATED TEXT")
print("-" * 80)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
print("-" * 80)
