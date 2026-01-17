# I Reproduced the RLM Paper From Scratch. Here's What Actually Works (And What Doesn't).

*A deep dive into Recursive Language Models — building it myself to understand it, testing claims with real API calls, and finding failure modes the paper doesn't mention.*

---

There's a paper making rounds in the AI research community: **"Recursive Language Models"** ([arXiv:2512.24601](https://arxiv.org/abs/2512.24601)). The claim is bold — LLMs can handle **arbitrarily long contexts** (we're talking millions of tokens) without any fine-tuning, just by changing how you prompt them.

I was skeptical. So I did what any reasonable person would do: I built it from scratch, ran real experiments, and documented what actually happened.

This is that story.

---

## The Problem RLM Tries to Solve

Here's the uncomfortable truth about modern LLMs: **they have amnesia beyond their context window**.

- Claude Sonnet: ~200K tokens
- GPT-4: ~128K tokens
- Most open models: 8K-32K tokens

If you have a million-character document and ask "find the secret code buried somewhere in here," the model literally cannot see most of your document. It gets truncated. The answer you need might be sitting at character 750,000 — but the model only ever sees the first 200,000.

This isn't a theoretical problem. Legal documents, codebases, research papers, financial reports — real-world data often exceeds these limits.

The standard solutions have trade-offs:
- **RAG (Retrieval)**: Chunk your document, embed chunks, retrieve "relevant" ones. Problem: the embedding model decides what's "relevant," and it might miss exactly what you need.
- **Fine-tuning on longer contexts**: Expensive, not always available, and still has limits.
- **Summarization**: You lose information. Period.

RLM proposes something different.

---

## The Core Idea (In Human Terms)

Imagine you're a detective searching a warehouse full of boxes for one specific document. You have two options:

**Option A (Direct Prompting):** Try to memorize the contents of every box simultaneously. When there are too many boxes, you just... forget the ones in the back.

**Option B (RLM):** Walk through the warehouse systematically. Open boxes, take notes, follow leads. You can't hold everything in your head, but you can write things down and keep searching.

RLM is Option B.

Instead of trying to fit the entire context into the LLM's limited memory, RLM:
1. Stores the full context in a **code execution environment** (a REPL)
2. Lets the LLM **write code** to explore that context
3. Feeds the code's output back to the LLM
4. Repeats until the LLM finds what it needs

The context exists as a variable. The LLM writes `context.find("secret code")` or `context[500000:600000]` to look at specific parts. It's programmatic exploration instead of trying to "see" everything at once.

---

## Why I Built It From Scratch

The paper has an [official repo](https://github.com/alexzhang13/rlm). I could have just run their code. But here's the thing — running someone else's code tells you *that* it works, not *why* it works.

I wanted to understand:
- What's the minimum viable implementation?
- Where does the "magic" actually happen?
- What breaks when you push it?

So I wrote ~1,700 lines of Python from scratch. No copy-pasting from the official repo. Just the paper, the concept, and a lot of debugging.

---

## The Architecture I Built

### Core Components

**1. The RLM Engine** (~350 lines)

The main loop looks like this:

```
User Query: "Find the secret magic number"
           ↓
┌──────────────────────────────────────┐
│           ROOT LLM                    │
│  Input: System prompt + query +       │
│         context metadata +            │
│         previous iteration results    │
│  Output: Text + code blocks           │
└──────────────────────────────────────┘
           ↓
┌──────────────────────────────────────┐
│           REPL                        │
│  Has: `context` variable (full doc)  │
│  Has: `llm_query()` function         │
│  Executes: Model's code blocks       │
│  Returns: stdout/stderr              │
└──────────────────────────────────────┘
           ↓
    Feed results back to LLM
           ↓
    Repeat until FINAL() called
```

The key insight: **the full context never gets sent to the LLM**. Only metadata ("you have 1 million characters of text") and code execution results get sent. The context lives in the REPL's memory.

**2. The REPL Environment** (~250 lines)

A sandboxed Python execution environment. The model can:
- Access `context` (the full document)
- Call `llm_query(prompt)` for sub-LLM queries
- Use standard Python: `len()`, `find()`, slicing, regex, etc.

The model cannot:
- Execute `eval()` or `exec()` on arbitrary strings
- Access the filesystem
- Make network calls (except through `llm_query`)

**3. The System Prompt** (~800 words)

This is where the magic happens. The system prompt teaches the model:
- That it has a REPL available
- How to write code blocks (` ```repl ... ``` `)
- Strategies for exploring large contexts (chunking, batched queries)
- How to signal completion (`FINAL(answer)`)

Without this prompt, the model has no idea what to do. With it, even a mid-tier model like GLM-4.7 can navigate million-character contexts.

---

## The Baselines I Compared Against

To know if RLM actually works, you need comparisons.

**1. Direct Prompting**
Just shove the whole context in the prompt. Simple. Fast. Breaks when context exceeds the window (it truncates and you lose data).

**2. RAG (Retrieval-Augmented Generation)**
Chunk the document, use BM25 to find "relevant" chunks based on the query, send only those chunks to the LLM. Problem: keyword matching misses semantic relevance.

**3. Chunked + Aggregate**
Process each chunk separately ("Is the answer in this chunk?"), then aggregate all chunk answers with a final LLM call. Problem: expensive (N LLM calls), and aggregation can introduce errors.

I implemented all three. Each is about 100-200 lines of Python.

---

## The Benchmarks

### Needle in a Haystack (NIAH)

The classic test: generate a massive document of random text, hide one "needle" (a specific fact), and see if the model can find it.

```python
# Generate 300,000 characters of random words
haystack = generate_random_text(300000)

# Insert needle at position 0.5 (150K characters in)
needle = "The secret magic number is 42857."
context = insert_at_position(haystack, needle, 0.5)

# Query
query = "What is the secret magic number?"
expected = "42857"
```

I varied:
- Context size: 100K, 300K, 500K, 1M characters
- Needle position: 0.1 (near start), 0.5 (middle), 0.9 (near end)
- Needle type: numbers, facts, code snippets

### Aggregation Tasks

Tests that require scanning the *entire* document:

**Counting:** "How many times is 'banana' mentioned across all sections?"

**Comparison:** "What's the highest revenue reported for Company X?"

LLMs are notoriously bad at counting. They hallucinate numbers. RLM can use `context.count('banana')` for an exact answer.

---

## The Results

I ran all experiments using **GLM-4.7** via the Anannas API. GLM-4.7 has a 205K token context window and costs $0.60 per million input tokens — cheap enough to run extensive tests.

### Experiment 1: Small Context (100K characters)

| Method | Correct | Time | Input Tokens |
|--------|---------|------|--------------|
| Direct | ✓ | 32.7s | 49,173 |
| RLM | ✓ | 129s | 3,449 |

**Both work.** Direct is faster. But notice the token count — RLM uses **14x fewer input tokens** because it doesn't send the full context each iteration.

### Experiment 2: Beyond the Window (300K characters)

Needle placed at position 0.9 (270K characters) — past the truncation point for Direct.

| Method | Correct | Time | Notes |
|--------|---------|------|-------|
| Direct | **✗** | 39s | Truncated at 200K, never saw the needle |
| RLM | ✓ | 90s | Found it in 3 iterations |

**This is where RLM earns its keep.** Direct literally cannot see the needle. RLM can.

### Experiment 3: Way Beyond (500K characters)

| Method | Correct | Time | Iterations |
|--------|---------|------|------------|
| Direct | **✗** | 51s | N/A |
| RLM | ✓ | 267s | 6 |

### Experiment 4: One Million Characters

Context: 999,904 characters (~250K tokens). Needle at position 0.75 (750K characters).

| Method | Correct | Time | Notes |
|--------|---------|------|-------|
| Direct | **✗** | 45s | Truncated, missed needle |
| RLM | ✓ | **14 minutes** | 6 iterations, found it |

**RLM handled a context 5x larger than the model's window.**

The model's strategy:
1. Explored context metadata (type, length)
2. Batched 5 sub-LLM queries across 200K chunks
3. Sub-LLM found nothing in first batch
4. Batched 5 more queries on different chunks
5. Sub-LLM found the needle in chunk 4
6. Returned `FINAL(28235)` ✓

### Experiment 5: Aggregation (Counting)

Task: Count occurrences of "banana" across 50 sections. Actual count: 18.

| Method | Answer | Correct |
|--------|--------|---------|
| Direct | 7 | **✗** |
| RLM | 18 | ✓ |

Direct **miscounted by 11**. This is typical — LLMs estimate, they don't count. RLM used `context.count('banana')` and got the exact answer.

---

## The Failure Modes (What the Paper Doesn't Emphasize)

Here's where it gets interesting. I ran edge case experiments to probe RLM's limits.

### Failure 1: The End-of-Context Blind Spot

I tested different needle positions in a 300K context:

| Position | RLM Result |
|----------|------------|
| 0.1 (30K) | ✓ |
| 0.3 (90K) | ✓ |
| 0.5 (150K) | ✓ |
| 0.7 (210K) | ✓ |
| **0.9 (270K)** | **✗ FAILED** |

Position 0.9 **failed**. The model's search strategy chunked from the beginning and ran out of iterations before reaching the end.

This is a real problem. If your needle is in the last 10% of a very long document, RLM might not find it.

### Failure 2: Structured Data Confusion

I tested random text vs. JSON:

| Format | RLM Result |
|--------|------------|
| Random words | ✓ |
| **JSON** | **✗ FAILED** |

The JSON context had many numeric values. RLM returned the wrong number — it got confused by the structure and picked a value that wasn't the needle.

### Failure 3: Won't Admit "Not Found"

I ran a test where the needle didn't exist. The expected behavior: "The answer is not in the context."

What RLM did: Described the context structure without admitting the answer wasn't there.

This is a hallucination risk. RLM won't say "I don't know."

### Failure 4: Multi-Hop Reasoning... Actually Works

I expected this to fail. Query: "What is the favorite color of the person who owns the most dogs?"

This requires:
1. Find who owns the most dogs (Diana: 7)
2. Find Diana's favorite color (purple)

RLM handled it in 6 iterations. Color me surprised.

---

## What I Learned

### The System Prompt Is Everything

The same model with a bad system prompt fails completely. With a good prompt, it becomes a methodical investigator. The prompt teaches *how* to use the tools, not just *that* tools exist.

### Token Efficiency Is Real

RLM doesn't send the full context every iteration — just code results. On a 500K context, Direct used ~100K input tokens per call. RLM used 7,660 across 6 iterations. That's **13x fewer tokens**.

### Latency Is The Cost

RLM is slow. 14 minutes for the 1M context test. Each iteration is a round-trip to the LLM API. If you need sub-second responses, this isn't for you.

### It's Inference-Only

No fine-tuning required. The same prompt works across models (I tested Claude Sonnet and GLM-4.7). This is significant — you're not locked into a specific model.

### The Real Innovation

The paper's contribution isn't "use a REPL" — that's been done. It's the realization that **storing context as a variable** instead of as prompt tokens fundamentally changes what's possible. The model becomes an agent exploring data, not a system trying to memorize it.

---

## When Should You Use RLM?

| Scenario | Recommendation |
|----------|----------------|
| Context fits in window | Direct prompting (faster, simpler) |
| Context exceeds window | **RLM (only option that works)** |
| Need exact counts/aggregation | RLM (code doesn't hallucinate) |
| Need low latency | Direct prompting |
| Cost-sensitive | RLM (fewer tokens overall) |
| Needle might be at the very end | Be careful — RLM has a blind spot |

---

## Reproducing This Yourself

Everything is on GitHub: [himanshud2611/hacks](https://github.com/himanshud2611/hacks)

The structure:
```
├── src/
│   ├── core/
│   │   ├── rlm.py         # Main RLM implementation
│   │   ├── repl.py        # Sandboxed REPL
│   │   └── llm_client.py  # API client
│   ├── benchmarks/
│   │   ├── niah.py        # Needle-in-haystack
│   │   └── aggregation.py # Counting/comparison
│   └── baselines/
│       ├── direct.py      # Direct prompting
│       ├── rag.py         # BM25 retrieval
│       └── chunked.py     # Chunk + aggregate
├── experiments/
│   └── run_exp.py         # Edge case tests
└── results/               # Raw JSON data
```

To run:
```bash
export ANANNAS_API_KEY="your-key"
python experiments/run_exp.py a b c d e f
```

---

## Final Thoughts

The RLM paper's claims are **mostly valid**, with caveats:

✓ **Handles contexts beyond the window** — Confirmed up to 1M characters  
✓ **Task-agnostic** — Same approach works for retrieval and aggregation  
✓ **Token-efficient** — 13-14x fewer input tokens than direct prompting  
⚠️ **Has blind spots** — Struggles with end-of-context needles  
⚠️ **Structured data issues** — JSON with many numbers confuses it  
⚠️ **Won't say "not found"** — Hallucination risk when answer doesn't exist  

Is RLM the future of long-context processing? Maybe. It's certainly clever. But it's not magic — it's a trade-off. You get unlimited context length at the cost of latency and some edge case failures.

For my use cases — analyzing large codebases, searching through documentation, processing research papers — it's worth the trade-off. Your mileage may vary.

---

## References

- Paper: [arXiv:2512.24601](https://arxiv.org/abs/2512.24601)
- Official Repo: [github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm)
- Author's Blogpost: [alexzhang13.github.io/blog/2025/rlm](https://alexzhang13.github.io/blog/2025/rlm/)
- My Reproduction: [github.com/himanshud2611/hacks](https://github.com/himanshud2611/hacks)

---

*If you found this useful, I occasionally write about AI systems and research reproduction. Reach out if you want to discuss.*
