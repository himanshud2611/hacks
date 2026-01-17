# RLM Paper Reproduction

**Reproducing "Recursive Language Models" (arXiv:2512.24601) from scratch**

## Overview

This repository contains a from-scratch implementation of the RLM (Recursive Language Model) inference strategy, along with comprehensive benchmarks and experiments.

## Key Findings

### ✅ What Works
- **1M character contexts** - RLM successfully handles contexts 5x larger than model window
- **Multi-hop reasoning** - Chains facts together correctly
- **Adversarial robustness** - Not fooled by decoy values
- **Aggregation tasks** - Accurate counting via code execution

### 🔴 Failure Modes Discovered
- **Position 0.9 blind spot** - Struggles to find needles at very end of long contexts
- **JSON format confusion** - Gets confused by structured data with many numbers
- **No "not found" capability** - Won't admit when answer doesn't exist

## Results Summary

| Context Size | Direct Baseline | RLM |
|--------------|-----------------|-----|
| 100K | ✓ | ✓ |
| 300K | ✗ (truncated) | ✓ |
| 500K | ✗ (truncated) | ✓ |
| 1M | ✗ (truncated) | ✓ |

## Structure

```
├── BENCHMARK.md           # Experiment tracking
├── FINDINGS.md            # Analysis report
├── RESEARCH_HYPOTHESES.md # Edge case experiments
├── src/
│   ├── core/
│   │   ├── rlm.py         # Main RLM implementation
│   │   ├── repl.py        # Sandboxed REPL environment
│   │   └── llm_client.py  # LLM client interfaces
│   ├── benchmarks/
│   │   ├── niah.py        # Needle-in-haystack
│   │   └── aggregation.py # Counting tasks
│   └── baselines/
│       ├── direct.py      # Direct prompting
│       ├── rag.py         # RAG baseline
│       └── chunked.py     # Chunk+aggregate
├── experiments/           # Edge case experiments
├── results/               # Raw JSON results
└── test_*.py              # Test scripts
```

## Usage

```python
from src.core.rlm import RLM
from src.core.anannas_client import AnannasClient

client = AnannasClient(model="zai-org/glm-4.7")
rlm = RLM(root_client=client, verbose=True)

result = rlm.completion(
    context="<your long context here>",
    query="What is the secret information?"
)
print(result.answer)
```

## References

- Paper: https://arxiv.org/abs/2512.24601
- Official Repo: https://github.com/alexzhang13/rlm
- Blogpost: https://alexzhang13.github.io/blog/2025/rlm/

## License

MIT
