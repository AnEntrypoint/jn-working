import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

print("=" * 70)
print("Jet-Nemotron Story Generation")
print("=" * 70)

MODEL_NAME = "jet-ai/Jet-Nemotron-2B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nModel: {MODEL_NAME}")
print(f"Device: {DEVICE}")

if DEVICE == "cpu":
    print("\n⚠ CUDA not available. Exiting...")
    exit(1)

print("\n" + "-" * 70)
print("Loading model and tokenizer...")
print("-" * 70)

start_time = time.time()

try:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    print("✓ Tokenizer loaded")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=DEVICE,
        attn_implementation="sdpa"
    )
    print(f"✓ Model loaded with SDPA attention ({time.time() - start_time:.2f}s)")

except Exception as e:
    import traceback
    print(f"✗ Error loading model: {e}")
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 70)
print("Generating Short Story (10 paragraphs)")
print("=" * 70)

story_prompt = """Write a short story in 10 paragraphs about a young engineer who discovers an ancient AI hidden deep within a forgotten server farm. The story should be engaging and mysterious.

Story:

"""

print(f"\nPrompt: {story_prompt.strip()}\n")
print("-" * 70)
print("Generating...")
print("-" * 70)

inputs = tokenizer(story_prompt, return_tensors="pt").to(DEVICE)

gen_start = time.time()

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=800,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )

gen_time = time.time() - gen_start

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
story_only = generated_text.replace(story_prompt, "").strip()

print("\n" + "=" * 70)
print("GENERATED STORY")
print("=" * 70)
print()
print(story_only)
print()
print("=" * 70)
print(f"Generation time: {gen_time:.2f}s")
print(f"Tokens generated: ~{len(outputs[0]) - len(inputs['input_ids'][0])}")
print(f"Tokens/second: ~{(len(outputs[0]) - len(inputs['input_ids'][0])) / gen_time:.1f}")
print("=" * 70)

mem_used = torch.cuda.max_memory_allocated() / 1024**3
print(f"\nPeak GPU memory: {mem_used:.2f} GB")
print("\n✓ Story generation complete!")
