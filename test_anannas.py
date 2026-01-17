"""
Real test with Anannas API
"""

import os
import sys
sys.path.insert(0, '/workspace/tools/python-packages')
sys.path.insert(0, '/workspace/himanshud-hacks/rlm-reproduction')

from src.benchmarks.niah import generate_niah_sample
from src.core.anannas_client import AnannasClient
from src.core.rlm import RLM

# Check for API key
api_key = os.environ.get("ANANNAS_API_KEY")
if not api_key:
    print("ERROR: ANANNAS_API_KEY not set")
    print("Run: export ANANNAS_API_KEY='your-key'")
    sys.exit(1)

print("=" * 70)
print("RLM TEST WITH ANANNAS API")
print("=" * 70)

# Generate test sample - small for quick test
sample = generate_niah_sample(
    target_length=5000,  # 5K chars - small test
    needle_position=0.5,
    needle_type="number",
    seed=42
)

print(f"\nTest Sample:")
print(f"  Context: {sample.context_length} chars")
print(f"  Query: {sample.query}")
print(f"  Expected: {sample.answer}")
print()

# Test with Claude 3.5 Sonnet via Anannas
MODEL = "anthropic/claude-3.5-sonnet"
print(f"Model: {MODEL}")
print("-" * 70)

try:
    client = AnannasClient(model=MODEL, api_key=api_key)
    
    # Quick API test first
    print("\n[1] Testing API connection...")
    test_response = client.completion([{"role": "user", "content": "Say 'API working' in exactly 2 words"}])
    print(f"    Response: {test_response[:50]}")
    print("    ✓ API connection works!")
    
    # Run RLM
    print("\n[2] Running RLM...")
    rlm = RLM(
        root_client=client,
        verbose=True,
        max_iterations=10,
    )
    
    result = rlm.completion(sample.context, sample.query)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Answer: {result.answer}")
    print(f"Expected: {sample.answer}")
    
    # Check correctness
    is_correct = sample.answer in result.answer
    print(f"Correct: {'✓ YES' if is_correct else '✗ NO'}")
    
    print(f"\nIterations: {len(result.iterations)}")
    print(f"Time: {result.total_time:.2f}s")
    print(f"Root tokens: {result.root_usage.input_tokens} in / {result.root_usage.output_tokens} out")
    print(f"Sub tokens: {result.sub_usage.input_tokens} in / {result.sub_usage.output_tokens} out")
    
    # Show iteration summary
    print("\n" + "-" * 70)
    print("ITERATION SUMMARY")
    print("-" * 70)
    for i, iteration in enumerate(result.iterations):
        print(f"  {i+1}. Code blocks: {len(iteration.code_blocks)}, Final: {iteration.has_final_answer}")
        if iteration.code_results:
            for j, r in enumerate(iteration.code_results):
                out = r.stdout[:80].replace('\n', ' ') if r.stdout else '(no output)'
                print(f"     Output {j+1}: {out}")

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
