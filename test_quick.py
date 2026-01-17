"""Quick test of RLM implementation."""

import sys
sys.path.insert(0, '/workspace/tools/python-packages')
sys.path.insert(0, '/workspace/himanshud-hacks/rlm-reproduction')

from src.benchmarks.niah import generate_niah_sample
from src.core.llm_client import AnthropicClient
from src.core.rlm import RLM

# Generate a small test sample
print("Generating NIAH sample...")
sample = generate_niah_sample(
    target_length=5000,  # Small for quick test
    needle_position=0.5,
    needle_type="number",
    seed=42
)

print(f"Context length: {sample.context_length}")
print(f"Query: {sample.query}")
print(f"Expected answer: {sample.answer}")
print(f"Needle: {sample.needle}")
print()

# Test with RLM
print("=" * 60)
print("Testing RLM...")
print("=" * 60)

try:
    client = AnthropicClient(model="claude-sonnet-4-20250514")
    rlm = RLM(root_client=client, verbose=True, max_iterations=10)
    
    result = rlm.completion(sample.context, sample.query)
    
    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Answer: {result.answer}")
    print(f"Expected: {sample.answer}")
    print(f"Correct: {sample.answer in result.answer}")
    print(f"Iterations: {len(result.iterations)}")
    print(f"Time: {result.total_time:.2f}s")
    print(f"Root tokens: {result.root_usage.input_tokens} in / {result.root_usage.output_tokens} out")
    print(f"Sub tokens: {result.sub_usage.input_tokens} in / {result.sub_usage.output_tokens} out")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
