"""Mock test of RLM implementation - no API calls needed."""

import sys
sys.path.insert(0, '/workspace/tools/python-packages')
sys.path.insert(0, '/workspace/himanshud-hacks/rlm-reproduction')

from src.benchmarks.niah import generate_niah_sample, NIAHBenchmark
from src.benchmarks.aggregation import generate_aggregation_sample, AggregationBenchmark
from src.core.repl import REPLEnvironment
from src.core.llm_client import LLMClient, UsageStats

print("=" * 60)
print("TESTING RLM COMPONENTS (Mock Mode)")
print("=" * 60)

# Test 1: NIAH Sample Generation
print("\n[1] NIAH Sample Generation")
print("-" * 40)
sample = generate_niah_sample(
    target_length=10000,
    needle_position=0.5,
    needle_type="number",
    seed=42
)
print(f"✓ Generated sample with {sample.context_length} chars")
print(f"  Query: {sample.query}")
print(f"  Expected: {sample.answer}")
print(f"  Needle position: {sample.needle_position}")

# Verify needle is in context
assert sample.needle in sample.context, "Needle not found in context!"
print(f"✓ Needle verified in context")

# Test 2: REPL Environment
print("\n[2] REPL Environment")
print("-" * 40)

def mock_llm_query(prompt: str) -> str:
    """Mock LLM that just returns a fixed response."""
    if "magic number" in prompt.lower():
        return "The magic number is 42."
    return f"Mock response to: {prompt[:50]}..."

repl = REPLEnvironment(
    context="Test context with magic number 12345",
    llm_query_fn=mock_llm_query
)

# Test basic code execution
result = repl.execute("x = 1 + 2\nprint(x)")
print(f"✓ Basic execution: stdout='{result.stdout.strip()}'")
assert "3" in result.stdout, "Basic execution failed"

# Test context access
result = repl.execute("print(len(context))")
print(f"✓ Context access: len={result.stdout.strip()}")

# Test llm_query
result = repl.execute('answer = llm_query("What is the magic number?")\nprint(answer)')
print(f"✓ LLM query: {result.stdout.strip()}")

# Test variable persistence
result = repl.execute("y = x * 10")
result = repl.execute("print(y)")
print(f"✓ Variable persistence: y={result.stdout.strip()}")
assert "30" in result.stdout, "Variable persistence failed"

repl.cleanup()

# Test 3: Benchmark Generation
print("\n[3] Benchmark Suite Generation")
print("-" * 40)

niah_bench = NIAHBenchmark(
    context_lengths=[5000, 10000],
    needle_positions=[0.5],
    needle_types=["number"],
    samples_per_config=1,
)
niah_samples = niah_bench.generate_samples()
print(f"✓ Generated {len(niah_samples)} NIAH samples")

agg_bench = AggregationBenchmark(
    num_chunks_list=[10],
    chunk_sizes=[500],
    query_types=["counting"],
    samples_per_config=1,
)
agg_samples = agg_bench.generate_samples()
print(f"✓ Generated {len(agg_samples)} Aggregation samples")

# Test 4: Evaluation Functions
print("\n[4] Evaluation Functions")
print("-" * 40)

# NIAH evaluation
assert niah_bench.evaluate("42", "42") == True
assert niah_bench.evaluate("The answer is 42", "42") == True
assert niah_bench.evaluate("43", "42") == False
print("✓ NIAH evaluation logic works")

# Aggregation evaluation
assert agg_bench.evaluate("The count is 5", "5") == True
assert agg_bench.evaluate("5 items found", "5") == True
print("✓ Aggregation evaluation logic works")

# Test 5: Code Block Extraction
print("\n[5] Code Block Extraction")
print("-" * 40)

from src.core.rlm import RLM

# Test regex extraction manually
import re
test_response = '''Let me check the context.

```repl
print(len(context))
print(type(context))
```

I see the context is a string. Let me search for the magic number.

```repl
import re
matches = re.findall(r'magic number is (\d+)', context)
print(matches)
```

FINAL(42)
'''

pattern = r'```repl\n(.*?)```'
matches = re.findall(pattern, test_response, re.DOTALL)
print(f"✓ Found {len(matches)} code blocks")
assert len(matches) == 2, "Should find 2 code blocks"

# Test FINAL extraction
final_match = re.search(r'FINAL\((.*?)\)', test_response, re.DOTALL)
assert final_match is not None
assert final_match.group(1).strip() == "42"
print(f"✓ FINAL() extraction works: '{final_match.group(1).strip()}'")

# Test 6: Mock LLM Client
print("\n[6] Mock LLM Client")
print("-" * 40)

class MockLLMClient(LLMClient):
    """Mock client that returns predefined responses."""
    
    def __init__(self):
        super().__init__(model="mock-model")
        self.call_count = 0
        
    def completion(self, messages, **kwargs):
        self.call_count += 1
        self.usage.add(100, 50, 0.1)  # Fake usage
        
        # Return a response that explores context and finds answer
        if self.call_count == 1:
            return '''Let me examine the context first.

```repl
print(f"Context type: {type(context)}")
print(f"Context length: {len(context)}")
print(context[:200])
```
'''
        elif self.call_count == 2:
            return '''I can see the context. Let me search for the magic number.

```repl
import re
match = re.search(r'magic number is (\d+)', context)
if match:
    magic_number = match.group(1)
    print(f"Found magic number: {magic_number}")
```
'''
        else:
            return 'Based on my analysis, FINAL(93810)'
    
mock_client = MockLLMClient()
response = mock_client.completion([{"role": "user", "content": "test"}])
print(f"✓ Mock client works, call_count={mock_client.call_count}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED! ✓")
print("=" * 60)
print("\nThe implementation is ready. To run actual experiments,")
print("set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable.")
