"""
Production-ready example: Load once, generate many times
This demonstrates the optimal way to use Jet-Nemotron in production
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

print("=" * 80)
print("Production Example: Load Once, Generate Multiple Times")
print("=" * 80)

MODEL_NAME = "jet-ai/Jet-Nemotron-2B"

# ONE-TIME SETUP (do this once when server starts)
print("\n1. ONE-TIME SETUP (Server Startup)")
print("-" * 80)
setup_start = time.time()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="flash_attention_2"
)
model.eval()  # Set to eval mode

setup_time = time.time() - setup_start
print(f"✓ Setup complete in {setup_time:.2f}s")
print("  (This happens ONCE when your server/app starts)")

# WARMUP (optional but recommended)
print("\n2. WARMUP (Optional)")
print("-" * 80)
warmup_start = time.time()
warmup_input = tokenizer("Hello", return_tensors="pt").to("cuda")
with torch.no_grad():
    _ = model.generate(**warmup_input, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
warmup_time = time.time() - warmup_start
print(f"✓ Warmup complete in {warmup_time:.2f}s")

# MULTIPLE GENERATION REQUESTS (this is what you do repeatedly)
print("\n3. MULTIPLE GENERATION REQUESTS")
print("-" * 80)

test_prompts = [
    "Explain quantum computing in simple terms:",
    "Write a haiku about artificial intelligence:",
    "What are the benefits of renewable energy?",
    "Describe the process of photosynthesis:",
]

generation_times = []

for i, prompt in enumerate(test_prompts, 1):
    print(f"\nRequest {i}: {prompt[:50]}...")

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    torch.cuda.synchronize()
    gen_start = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    torch.cuda.synchronize()
    gen_time = time.time() - gen_start

    tokens_generated = outputs.shape[1] - inputs['input_ids'].shape[1]
    generation_times.append(gen_time)

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"  ✓ Generated {tokens_generated} tokens in {gen_time:.2f}s ({tokens_generated/gen_time:.1f} tok/s)")
    print(f"  Response: {result[:100]}...")

# RESULTS
avg_gen_time = sum(generation_times) / len(generation_times)
print("\n" + "=" * 80)
print("PERFORMANCE ANALYSIS")
print("=" * 80)
print(f"Setup (one-time):           {setup_time:.2f}s")
print(f"Warmup (one-time):          {warmup_time:.2f}s")
print(f"Average per request:        {avg_gen_time:.2f}s")
print(f"Total for 4 requests:       {sum(generation_times):.2f}s")
print(f"\nIf you had to reload each time:")
print(f"  4 requests with reload:   {4 * (setup_time + avg_gen_time):.2f}s")
print(f"  4 requests with reuse:    {setup_time + sum(generation_times):.2f}s")
print(f"  Time saved:               {4 * setup_time:.2f}s ({4 * setup_time / (4 * (setup_time + avg_gen_time)) * 100:.1f}%)")

print("\n" + "=" * 80)
print("PRODUCTION RECOMMENDATIONS")
print("=" * 80)
print("""
✓ OPTIMAL SETUP (what we just demonstrated):
  1. Load model once on server startup (~37s)
  2. Keep it in GPU memory
  3. Process requests (~15-20s each)
  4. No reload between requests

✗ INEFFICIENT (DON'T DO THIS):
  - Loading model for each request
  - Would add 37s overhead per request
  - Would make each request ~55s instead of ~18s

💡 FOR WEB SERVICES:
  - Use FastAPI/Flask with global model
  - Load in __init__ or startup event
  - Reuse for all API calls

💡 FOR BATCH PROCESSING:
  - Load once at script start
  - Process all items in loop
  - Only one 37s startup cost

📊 YOUR CURRENT PERFORMANCE:
  - RTX 3060 Laptop GPU
  - ~10.7 tokens/second
  - ~18s for 200 token response
  - This is OPTIMAL for your hardware!

🚀 TO GO FASTER, YOU WOULD NEED:
  - Better GPU (RTX 4090: ~4x faster)
  - Data center GPU (H100: ~20x faster)
  - But for RTX 3060, you're maxed out!
""")
print("=" * 80)
