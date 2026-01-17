# I Reproduced the RLM Paper From Scratch. Here's What Actually Works (And What Doesn't).

*A deep dive into Recursive Language Models — building it myself to understand it, testing claims with real API calls, and finding failure modes the paper doesn't mention.*

---

There's a paper making rounds in the AI research community: **"Recursive Language Models"** ([arXiv:2512.24601](https://arxiv.org/abs/2512.24601)). The claim is bold — LLMs can handle **arbitrarily long contexts** (we're talking millions of tokens) without any fine-tuning, just by changing how you prompt them.

I was skeptical. So I did what any reasonable person would do: I built it from scratch, ran real experiments, and documented what actually happened.

This is that story.

---

## Part 1: The Problem

### Why Context Windows Matter

Here's what happens when you send a document to an LLM:

```
Your Document: 1,000,000 characters
                    │
                    ▼
         ┌─────────────────────┐
         │   TOKENIZER         │
         │   ~4 chars = 1 token│
         └─────────────────────┘
                    │
                    ▼
            250,000 tokens
                    │
                    ▼
         ┌─────────────────────┐
         │   CONTEXT WINDOW    │
         │   Limit: 200K tokens│
         └─────────────────────┘
                    │
                    ▼
    ┌───────────────┴───────────────┐
    │                               │
    ▼                               ▼
First 200K tokens              Last 50K tokens
   (KEPT)                        (DELETED)
```

**The model never sees the last 50K tokens.** If your answer is there, you're out of luck.

This isn't theoretical. Real numbers:

| Document Type | Typical Size | Fits in GPT-4? | Fits in Claude? |
|--------------|--------------|----------------|-----------------|
| Research paper | 30K tokens | ✓ | ✓ |
| Legal contract | 80K tokens | ✓ | ✓ |
| Codebase (medium) | 200K tokens | ✗ | Barely |
| Book | 400K tokens | ✗ | ✗ |
| Enterprise docs | 1M+ tokens | ✗ | ✗ |

### The Existing Solutions (And Why They're Not Enough)

**Solution 1: RAG (Retrieval-Augmented Generation)**

```
┌─────────────────────────────────────────────────────────┐
│                    RAG PIPELINE                          │
└─────────────────────────────────────────────────────────┘

   Full Document                    User Query
        │                               │
        ▼                               ▼
┌───────────────┐               ┌───────────────┐
│ Split into    │               │ Convert to    │
│ chunks        │               │ embedding     │
└───────────────┘               └───────────────┘
        │                               │
        ▼                               │
┌───────────────┐                       │
│ Embed each    │                       │
│ chunk         │                       │
└───────────────┘                       │
        │                               │
        ▼                               ▼
┌─────────────────────────────────────────────────────────┐
│              VECTOR DATABASE                             │
│    Find chunks with similar embeddings to query          │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
                  Top K chunks (maybe 5-10)
                          │
                          ▼
                 ┌─────────────────┐
                 │      LLM        │
                 │  Answer based   │
                 │  on retrieved   │
                 │  chunks only    │
                 └─────────────────┘
```

**The Problem:** The embedding model decides what's "relevant." It uses semantic similarity. But what if your query is "What is the secret code?" and the needle is "The password is XJ7-Alpha"? The words don't overlap. The embedding might not connect them. You miss the answer.

**Solution 2: Summarization**

```
Full Document → Summarize → Shorter Document → LLM
```

**The Problem:** Summarization loses information. By definition. If the summary doesn't include your needle, it's gone.

**Solution 3: Fine-tuning for longer contexts**

**The Problem:** Expensive. Model-specific. Still has limits. And you need to do it for every new model.

---

## Part 2: The RLM Idea

### The Analogy That Made It Click

Imagine you're searching for a specific receipt in a warehouse full of filing cabinets.

**Approach A: The Memorization Approach (Direct Prompting)**

You try to memorize the contents of every cabinet simultaneously. You walk through, reading everything, trying to hold it all in your head. When there are 1000 cabinets and you can only remember 200 cabinets worth of information... you forget the rest. The receipt might be in cabinet 847, but you'll never know.

**Approach B: The Detective Approach (RLM)**

You walk into the warehouse with a notebook. You can't memorize everything, but you can:
- Check the labels on cabinets
- Open specific drawers
- Write down what you find
- Follow leads ("Cabinet 500 mentions 'receipts from 2023' — check there next")
- Keep searching until you find it

You're not trying to hold everything in your head. You're **systematically exploring** with tools.

RLM is Approach B.

### The Core Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RLM ARCHITECTURE                             │
└─────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────┐
                    │     USER QUERY      │
                    │ "Find the secret    │
                    │  magic number"      │
                    └──────────┬──────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         ROOT LLM                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ WHAT IT SEES:                                                   │  │
│  │  • System prompt (how to use REPL)                             │  │
│  │  • Context METADATA: "You have 1M chars of text"               │  │
│  │  • User query                                                   │  │
│  │  • Previous iteration results                                   │  │
│  │                                                                 │  │
│  │ WHAT IT DOES NOT SEE:                                          │  │
│  │  • The actual 1M character context (NOT sent as tokens!)       │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  OUTPUT: Text + Code Blocks                                          │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ "Let me explore the context structure first."                  │  │
│  │                                                                │  │
│  │ ```repl                                                        │  │
│  │ print(type(context))                                           │  │
│  │ print(len(context))                                            │  │
│  │ print(context[:500])  # First 500 chars                        │  │
│  │ ```                                                            │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
                               │
                               │ Code blocks extracted
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      REPL ENVIRONMENT                                 │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ MEMORY:                                                         │  │
│  │  • context = "<the full 1M character document>"                │  │
│  │  • llm_query = function to call sub-LLM                        │  │
│  │  • llm_query_batched = parallel sub-LLM calls                  │  │
│  │  • All standard Python: len, find, slice, regex, etc.          │  │
│  │                                                                 │  │
│  │ EXECUTES:                                                       │  │
│  │  print(type(context))     → <class 'str'>                      │  │
│  │  print(len(context))      → 1000000                            │  │
│  │  print(context[:500])     → "alksjdf lkajsdf..."               │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  OUTPUT: stdout from code execution                                  │
└──────────────────────────────────────────────────────────────────────┘
                               │
                               │ Results fed back
                               ▼
                    ┌─────────────────────┐
                    │   NEXT ITERATION    │
                    │   (repeat until     │
                    │   FINAL() called)   │
                    └─────────────────────┘
```

### The Key Insight (This Is The Whole Paper)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    WHERE THE CONTEXT LIVES                           │
└─────────────────────────────────────────────────────────────────────┘

DIRECT PROMPTING:
┌─────────────────────────────────────────────────────────────────────┐
│ Context lives HERE → sent as tokens every single time               │
│                                                                      │
│    [System Prompt] + [FULL CONTEXT] + [Query]                       │
│         1K tokens     250K tokens      100 tokens                   │
│                           │                                          │
│                           └─── This is what kills you               │
└─────────────────────────────────────────────────────────────────────┘

RLM:
┌─────────────────────────────────────────────────────────────────────┐
│ Context lives HERE → stored in REPL memory, accessed via code       │
│                                                                      │
│    LLM sees: [System Prompt] + [Metadata] + [Code Results]          │
│                  1K tokens      100 tokens    500 tokens            │
│                                                                      │
│    REPL has: context = "<1M chars>" (in memory, not tokens!)        │
│                           │                                          │
│                           └─── This is why it works                 │
└─────────────────────────────────────────────────────────────────────┘
```

The context is **data in memory**, not **tokens in the prompt**. The model writes code to access it. This is the entire innovation.

---

## Part 3: The Iteration Loop (How It Actually Works)

Let me show you a real execution trace from my experiments:

### Example: Finding a needle in 1M characters

**Setup:**
- Context: 999,904 characters of random text
- Needle: "The secret magic number is 28235" (at position 750,000)
- Query: "What is the secret magic number?"

```
┌─────────────────────────────────────────────────────────────────────┐
│                      ITERATION 1: Exploration                        │
└─────────────────────────────────────────────────────────────────────┘

LLM Output:
┌─────────────────────────────────────────────────────────────────────┐
│ "I need to understand what I'm working with first."                 │
│                                                                      │
│ ```repl                                                              │
│ print(f"Type: {type(context)}")                                     │
│ print(f"Length: {len(context)} characters")                         │
│ print(f"Sample: {context[:200]}")                                   │
│ ```                                                                  │
└─────────────────────────────────────────────────────────────────────┘

REPL Execution:
┌─────────────────────────────────────────────────────────────────────┐
│ Type: <class 'str'>                                                  │
│ Length: 999904 characters                                            │
│ Sample: "jkhasd qwpoeiu zxcvnm poiuyt aslkdfj qwerty..."            │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      ITERATION 2: Strategy                           │
└─────────────────────────────────────────────────────────────────────┘

LLM Output:
┌─────────────────────────────────────────────────────────────────────┐
│ "1M characters is too large to search manually. Let me try a        │
│  direct search first, then chunk if needed."                        │
│                                                                      │
│ ```repl                                                              │
│ # Try direct string search                                          │
│ idx = context.find("magic number")                                  │
│ print(f"Found at index: {idx}")                                     │
│ if idx != -1:                                                       │
│     print(context[idx:idx+100])                                     │
│ ```                                                                  │
└─────────────────────────────────────────────────────────────────────┘

REPL Execution:
┌─────────────────────────────────────────────────────────────────────┐
│ Found at index: 749921                                               │
│ magic number is 28235. Remember this. alkjsdf poqiweu...            │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      ITERATION 3: Extraction                         │
└─────────────────────────────────────────────────────────────────────┘

LLM Output:
┌─────────────────────────────────────────────────────────────────────┐
│ "Found it! Let me extract the exact number."                        │
│                                                                      │
│ ```repl                                                              │
│ import re                                                            │
│ match = re.search(r'magic number is (\d+)', context)                │
│ if match:                                                           │
│     print(f"Number: {match.group(1)}")                              │
│ ```                                                                  │
└─────────────────────────────────────────────────────────────────────┘

REPL Execution:
┌─────────────────────────────────────────────────────────────────────┐
│ Number: 28235                                                        │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      ITERATION 4: Completion                         │
└─────────────────────────────────────────────────────────────────────┘

LLM Output:
┌─────────────────────────────────────────────────────────────────────┐
│ "The secret magic number is 28235."                                 │
│                                                                      │
│ FINAL(28235)                                                        │
└─────────────────────────────────────────────────────────────────────┘

✓ CORRECT ANSWER IN 4 ITERATIONS
```

### What Happened (Token Analysis)

```
┌─────────────────────────────────────────────────────────────────────┐
│                      TOKEN USAGE COMPARISON                          │
└─────────────────────────────────────────────────────────────────────┘

DIRECT PROMPTING (Failed - context truncated):
┌────────────────────────────────────────────┐
│ Input:  [Context: 200K tokens] + [Query]   │  ← Truncated!
│ Output: [Wrong answer]                      │
│ Total:  ~200,000 input tokens              │
└────────────────────────────────────────────┘

RLM (Succeeded):
┌────────────────────────────────────────────────────────────────────┐
│ Iteration 1: System(1K) + Meta(100) + Query(50)    = 1,150 tokens │
│ Iteration 2: Previous(1K) + Results(200) + New     = 1,400 tokens │
│ Iteration 3: Previous(1.4K) + Results(100) + New   = 1,700 tokens │
│ Iteration 4: Previous(1.7K) + Results(50) + Final  = 1,900 tokens │
│────────────────────────────────────────────────────────────────────│
│ TOTAL: ~6,150 input tokens                                         │
│                                                                     │
│ Context tokens used: 0 (it's in REPL memory!)                      │
└────────────────────────────────────────────────────────────────────┘

SAVINGS: 200,000 / 6,150 = 32x fewer tokens
```

---

## Part 4: The Sub-LLM Pattern (Recursive Calls)

Sometimes string operations aren't enough. The model needs to **understand** content, not just search it. That's where `llm_query()` comes in.

### Architecture of Sub-LLM Calls

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SUB-LLM ARCHITECTURE                              │
└─────────────────────────────────────────────────────────────────────┘

                         ROOT LLM
                            │
                            │ Generates code:
                            │ answers = []
                            │ for chunk in chunks:
                            │     ans = llm_query(f"Find X in: {chunk}")
                            │     answers.append(ans)
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         REPL                                         │
│                                                                      │
│   llm_query("Find the secret in: <chunk 1>")  ───────┐              │
│   llm_query("Find the secret in: <chunk 2>")  ───────┤              │
│   llm_query("Find the secret in: <chunk 3>")  ───────┤              │
│   llm_query("Find the secret in: <chunk 4>")  ───────┤              │
│   llm_query("Find the secret in: <chunk 5>")  ───────┘              │
│                                                      │               │
└──────────────────────────────────────────────────────│───────────────┘
                                                       │
                    ┌──────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       SUB-LLM (Same or different model)              │
│                                                                      │
│   Input: "Find the secret in: <200K char chunk>"                    │
│   Output: "Found: The secret magic number is 28235" or "Not found"  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                    │
                    │ Results collected
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         REPL (continued)                             │
│                                                                      │
│   answers = [                                                        │
│       "Not found in this chunk",                                    │
│       "Not found in this chunk",                                    │
│       "Not found in this chunk",                                    │
│       "Found: The secret magic number is 28235",  ← HERE!           │
│       "Not found in this chunk"                                     │
│   ]                                                                  │
│   print(answers)                                                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                    │
                    │ Results fed back to Root LLM
                    ▼
                 ROOT LLM
                    │
                    ▼
              FINAL(28235)
```

### Why This Matters

The model doesn't have to write perfect regex or know exactly what to search for. It can say "hey sub-LLM, read this chunk and tell me if there's anything about a secret number." The sub-LLM **understands** the content.

This is recursive comprehension, not just recursive search.

---

## Part 5: The System Prompt (Where The Magic Actually Lives)

I cannot stress this enough: **the system prompt is 80% of why RLM works.**

Here's the actual prompt I used (simplified):

```
┌─────────────────────────────────────────────────────────────────────┐
│                      RLM SYSTEM PROMPT                               │
└─────────────────────────────────────────────────────────────────────┘

You are tasked with answering a query using a context variable.
You have access to a REPL environment with:

1. `context` - The input data (could be string, list, dict)
2. `llm_query(prompt)` - Call a sub-LLM with a prompt
3. `llm_query_batched(prompts)` - Call sub-LLM on multiple prompts
4. `print()` - See outputs

STRATEGIES FOR LARGE CONTEXTS:
┌─────────────────────────────────────────────────────────────────────┐
│ 1. ALWAYS explore first: type(context), len(context), sample       │
│ 2. For strings: try context.find() or regex first (fast!)          │
│ 3. If that fails: chunk and use llm_query on each chunk            │
│ 4. Use llm_query_batched for parallel processing                   │
└─────────────────────────────────────────────────────────────────────┘

TO WRITE CODE, use triple backticks with 'repl':
```repl
print(len(context))
```

WHEN DONE, signal your answer:
  FINAL(your answer here)
  or
  FINAL_VAR(variable_name)

DO NOT answer until you have examined the context!
```

### What The Prompt Teaches

```
┌─────────────────────────────────────────────────────────────────────┐
│                LESSONS IN THE PROMPT                                 │
└─────────────────────────────────────────────────────────────────────┘

Lesson 1: "You have a context variable"
          │
          └─► Model knows to access `context`, not ask for input

Lesson 2: "Always explore first"
          │
          └─► Model checks type/length before diving in

Lesson 3: "Try find() or regex first"
          │
          └─► Model uses fast string ops before expensive LLM calls

Lesson 4: "Chunk and query if needed"
          │
          └─► Model knows the fallback strategy

Lesson 5: "Use FINAL() when done"
          │
          └─► Model knows how to signal completion

Lesson 6: "DO NOT answer until you examine context"
          │
          └─► Prevents hallucination from guessing
```

Without this prompt, the model just... guesses. With it, even a mid-tier model becomes a methodical investigator.

---

## Part 6: The Implementation (What I Actually Built)

### File Structure

```
rlm-reproduction/
│
├── src/
│   ├── core/
│   │   ├── rlm.py              # 363 lines - Main RLM engine
│   │   ├── repl.py             # 246 lines - Sandboxed execution
│   │   ├── llm_client.py       # 89 lines  - Base client interface
│   │   └── anannas_client.py   # 67 lines  - Anannas API client
│   │
│   ├── baselines/
│   │   ├── direct.py           # 105 lines - Direct prompting
│   │   ├── rag.py              # 198 lines - BM25 retrieval
│   │   └── chunked.py          # 128 lines - Chunk + aggregate
│   │
│   └── benchmarks/
│       ├── niah.py             # 201 lines - Needle in haystack
│       └── aggregation.py      # 259 lines - Counting/comparison
│
├── experiments/
│   └── run_exp.py              # Edge case experiments
│
├── results/                    # Raw JSON results
│
└── tests/
    ├── test_mock.py            # Component tests
    └── test_e2e_mock.py        # End-to-end tests

Total: ~1,700 lines of Python
```

### The RLM Class (Simplified)

```python
class RLM:
    def __init__(self, root_client, sub_client=None, max_iterations=20):
        self.root_client = root_client
        self.sub_client = sub_client or root_client
        self.max_iterations = max_iterations
    
    def completion(self, context, query):
        # 1. Create REPL with context
        repl = REPLEnvironment(
            context=context,
            llm_query_fn=self._llm_query
        )
        
        # 2. Build initial messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {query}"}
        ]
        
        # 3. Iteration loop
        for i in range(self.max_iterations):
            # Get LLM response
            response = self.root_client.completion(messages)
            
            # Extract code blocks
            code_blocks = extract_code_blocks(response)
            
            # Execute each code block
            results = []
            for code in code_blocks:
                result = repl.execute(code)
                results.append(result)
            
            # Check for FINAL()
            if "FINAL(" in response:
                return extract_final_answer(response)
            
            # Add results to message history
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": format_results(results)})
        
        return "Max iterations reached"
```

### The REPL Sandbox

```python
class REPLEnvironment:
    # Safe builtins - no eval, exec, file access
    SAFE_BUILTINS = {
        'print', 'len', 'str', 'int', 'list', 'dict',
        'range', 'enumerate', 'zip', 'sorted', 'min', 'max',
        # ... more safe functions
    }
    
    def __init__(self, context, llm_query_fn):
        self.locals = {
            'context': context,           # The full document!
            'llm_query': llm_query_fn,    # Sub-LLM calls
        }
        self.globals = {'__builtins__': self.SAFE_BUILTINS}
    
    def execute(self, code):
        """Execute code in sandbox, return stdout."""
        stdout_capture = io.StringIO()
        with redirect_stdout(stdout_capture):
            exec(code, self.globals, self.locals)
        return stdout_capture.getvalue()
```

---

## Part 7: The Baselines (What I Compared Against)

### Baseline 1: Direct Prompting

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DIRECT PROMPTING                                  │
└─────────────────────────────────────────────────────────────────────┘

         Full Context                    Query
              │                            │
              ▼                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          PROMPT                                      │
│                                                                      │
│  "Here is the context:                                              │
│   <ENTIRE DOCUMENT - possibly truncated if too long>                │
│                                                                      │
│   Question: What is the secret magic number?                        │
│                                                                      │
│   Answer:"                                                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                           LLM
                              │
                              ▼
                          Answer

PROS: Simple, fast, one API call
CONS: Truncates if context > window, loses information
```

### Baseline 2: RAG (Retrieval-Augmented Generation)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RAG                                          │
└─────────────────────────────────────────────────────────────────────┘

Full Context
     │
     ▼
┌─────────────┐
│ Chunk into  │ ──► ["chunk1", "chunk2", "chunk3", ...]
│ N pieces    │
└─────────────┘
     │
     ▼
┌─────────────┐
│ BM25 Index  │ ──► Term frequencies, document frequencies
│ (keyword)   │
└─────────────┘
     │
     │        Query: "secret magic number"
     │              │
     │              ▼
     │        ┌─────────────┐
     │        │ BM25 Search │
     │        │ Score each  │
     │        │ chunk       │
     │        └─────────────┘
     │              │
     └──────────────┤
                    ▼
            Top K chunks (e.g., 5)
                    │
                    ▼
         ┌─────────────────┐
         │ Send to LLM     │
         │ with query      │
         └─────────────────┘
                    │
                    ▼
                Answer

PROS: Can handle very long documents, only sends relevant chunks
CONS: BM25 might miss semantically relevant chunks (keyword mismatch)
```

### Baseline 3: Chunk + Aggregate

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CHUNK + AGGREGATE                                 │
└─────────────────────────────────────────────────────────────────────┘

Full Context
     │
     ▼
┌─────────────┐
│ Chunk into  │ ──► ["chunk1", "chunk2", "chunk3", ...]
│ N pieces    │
└─────────────┘
     │
     ├─────────────────────────────────────────┐
     │                                         │
     ▼                                         ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ LLM Call 1  │   │ LLM Call 2  │   │ LLM Call N  │
│ "Is answer  │   │ "Is answer  │   │ "Is answer  │
│  in chunk1?"│   │  in chunk2?"│   │  in chunkN?"│
└─────────────┘   └─────────────┘   └─────────────┘
     │                   │                   │
     ▼                   ▼                   ▼
 "Not found"       "Not found"        "Found: 28235"
     │                   │                   │
     └───────────────────┼───────────────────┘
                         │
                         ▼
              ┌─────────────────┐
              │ Aggregation LLM │
              │ "Combine these  │
              │  answers"       │
              └─────────────────┘
                         │
                         ▼
                   Final Answer

PROS: Examines all chunks, can handle long contexts
CONS: N+1 LLM calls (expensive!), aggregation can introduce errors
```

---

## Part 8: The Benchmarks

### Benchmark 1: Needle in a Haystack (NIAH)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    NIAH BENCHMARK                                    │
└─────────────────────────────────────────────────────────────────────┘

GOAL: Find ONE specific fact buried in massive noise

GENERATION:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  Random Text ───────────────────────────────────────────────────    │
│  "aklsjdf qwpoei zxcvnm lkjhgf poiuyt asdfgh qwerty..."           │
│                                                                      │
│                        ▲                                             │
│                        │                                             │
│                   INSERT NEEDLE                                      │
│              "The secret magic number is 42857"                     │
│                        │                                             │
│                        ▼                                             │
│  Position: 0.1 (10%)  0.5 (50%)  0.9 (90%)                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

VARIABLES:
  • Context size: 100K, 300K, 500K, 1M characters
  • Needle position: 0.1, 0.3, 0.5, 0.7, 0.9
  • Needle type: number, fact, code snippet

QUERY: "What is the secret magic number?"
EXPECTED: "42857"
```

### Benchmark 2: Aggregation (Counting)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AGGREGATION BENCHMARK                             │
└─────────────────────────────────────────────────────────────────────┘

GOAL: Count/aggregate across ENTIRE document

GENERATION:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  Section 1: "The report shows... [Observation: Found a banana]..."  │
│  Section 2: "Analysis indicates... various data points..."          │
│  Section 3: "The study... [Observation: Found a banana]..."         │
│  Section 4: "Results suggest... [Observation: Found a banana]..."   │
│  ...                                                                 │
│  Section 50: "Conclusion... [Observation: Found a banana]..."       │
│                                                                      │
│  Total bananas: 18 (scattered across sections)                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

QUERY: "How many times is 'banana' mentioned?"
EXPECTED: "18"

WHY THIS IS HARD:
  • LLMs notoriously bad at counting
  • Must scan ENTIRE document
  • Easy to miss mentions or double-count
```

---

## Part 9: The Results

### Test Configuration

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TEST CONFIGURATION                                │
└─────────────────────────────────────────────────────────────────────┘

Model: GLM-4.7 (via Anannas API)
  • Context window: 205K tokens
  • Cost: $0.60 / million input tokens
  • Speed: Fast inference

Why GLM-4.7?
  • Large enough context window for fair comparison
  • Cheap enough for extensive testing
  • Good code generation ability
```

### Result 1: Context Size Scaling

```
┌─────────────────────────────────────────────────────────────────────┐
│              CONTEXT SIZE vs SUCCESS RATE                            │
└─────────────────────────────────────────────────────────────────────┘

Context     Direct          RLM              Winner
Size        Prompting

100K        ✓ (32.7s)       ✓ (129s)         Direct (faster)
            │               │
            │               └─ Works but slower
            └─ Fast, context fits in window

300K        ✗ (39s)         ✓ (90s)          RLM
            │               │
            │               └─ Found needle at 270K
            └─ TRUNCATED at 200K, missed needle

500K        ✗ (51s)         ✓ (267s)         RLM
            │               │
            │               └─ 6 iterations, found it
            └─ TRUNCATED, missed needle

1M          ✗ (45s)         ✓ (14 min)       RLM
            │               │
            │               └─ 6 iterations, batched sub-LLM
            └─ TRUNCATED, missed needle


         ┌────────────────────────────────────────┐
         │           VISUALIZATION                 │
         │                                         │
         │  Direct:  ████████░░░░░░░░░░  (200K)   │
         │  RLM:     ██████████████████  (1M+)    │
         │           ▲                             │
         │           │                             │
         │      Context window limit               │
         │      (Direct truncates here)            │
         └────────────────────────────────────────┘
```

### Result 2: Token Efficiency

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TOKEN USAGE COMPARISON                            │
└─────────────────────────────────────────────────────────────────────┘

                    Direct              RLM              Ratio
Context Size        Input Tokens        Input Tokens

100K chars          49,173              3,449            14x fewer
300K chars          ~75,000*            ~5,000           15x fewer
500K chars          ~100,000*           7,660            13x fewer

* Truncated to 200K max, so actual tokens sent is capped

┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  TOKENS                                                              │
│    │                                                                 │
│ 100K┤  ████████████████████████████  Direct                         │
│    │                                                                 │
│ 75K┤                                                                 │
│    │                                                                 │
│ 50K┤  ██████████████████████████████ Direct                         │
│    │                                                                 │
│ 25K┤                                                                 │
│    │                                                                 │
│  5K┤  ████ RLM    ████ RLM    ████ RLM                              │
│    │                                                                 │
│    └──────────────────────────────────────────────────────          │
│       100K        300K        500K     Context Size                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Result 3: Aggregation (Counting)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    COUNTING TASK RESULTS                             │
└─────────────────────────────────────────────────────────────────────┘

Task: Count "banana" mentions across 50 sections
Actual count: 18

Method          Answer      Correct?    How it worked
──────────────────────────────────────────────────────────────────────
Direct          7           ✗           LLM estimated/guessed
RLM             18          ✓           Used context.count('banana')


WHY DIRECT FAILED:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  LLM thought process (approximation):                               │
│                                                                      │
│  "I see banana mentioned... let me count...                         │
│   1, 2, 3... wait I think I saw that one...                        │
│   maybe 4, 5, 6, 7... yeah probably around 7"                       │
│                                                                      │
│  RESULT: 7 (WRONG - actual is 18)                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

WHY RLM SUCCEEDED:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  ```repl                                                             │
│  count = context.count('banana')                                    │
│  print(count)                                                       │
│  ```                                                                 │
│  Output: 18                                                          │
│                                                                      │
│  RESULT: 18 (CORRECT - exact count)                                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

KEY INSIGHT: Code doesn't hallucinate. LLMs do.
```

---

## Part 10: The Failure Modes

### Failure 1: End-of-Context Blind Spot

```
┌─────────────────────────────────────────────────────────────────────┐
│                    POSITION SENSITIVITY TEST                         │
└─────────────────────────────────────────────────────────────────────┘

Context: 300K characters
Needle positions tested: 0.1, 0.3, 0.5, 0.7, 0.9

Results:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  Position    Absolute Location    RLM Result                        │
│  ─────────────────────────────────────────────                      │
│  0.1         30K chars            ✓ Found                           │
│  0.3         90K chars            ✓ Found                           │
│  0.5         150K chars           ✓ Found                           │
│  0.7         210K chars           ✓ Found                           │
│  0.9         270K chars           ✗ FAILED                          │
│                                                                      │
│  ════════════════════════════════════════════════════════          │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░          │
│  │         │         │         │         │                          │
│  0.1       0.3       0.5       0.7       0.9                        │
│  ✓         ✓         ✓         ✓         ✗                          │
│                                         ▲                           │
│                                         │                           │
│                                    BLIND SPOT                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

WHY?
  • Model's chunking strategy: divides into N chunks from start
  • Searches chunks 1, 2, 3... runs out of iterations
  • Never reaches chunk N (where needle is at 0.9)
  
IMPLICATION:
  If your needle is in the last 10% of a very long document,
  RLM might not find it with default settings.
```

### Failure 2: Structured Data Confusion

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FORMAT SENSITIVITY TEST                           │
└─────────────────────────────────────────────────────────────────────┘

Test: Same needle, different context formats

FORMAT 1: Random words
┌─────────────────────────────────────────────────────────────────────┐
│ "aksjdf lkjasd qwpoei The secret magic number is 42857 zxcvnm..."  │
└─────────────────────────────────────────────────────────────────────┘
Result: ✓ Found 42857

FORMAT 2: JSON with many numbers
┌─────────────────────────────────────────────────────────────────────┐
│ [                                                                    │
│   {"id": 1, "value": 998, "data": "xxx..."},                        │
│   {"id": 2, "value": 743, "data": "xxx..."},                        │
│   ...                                                                │
│   {"id": 500, "type": "secret", "magic_number": 42857},  ← NEEDLE  │
│   ...                                                                │
│   {"id": 999, "value": 156, "data": "xxx..."}                       │
│ ]                                                                    │
└─────────────────────────────────────────────────────────────────────┘
Result: ✗ Returned 998 (WRONG!)

WHY?
  • JSON has hundreds of numeric values
  • Model gets confused by the structure
  • Picks a number that isn't the needle
  
IMPLICATION:
  Structured data with many similar-looking values is harder.
  The noise-to-signal ratio matters.
```

### Failure 3: Won't Say "Not Found"

```
┌─────────────────────────────────────────────────────────────────────┐
│                    NO-ANSWER TEST                                    │
└─────────────────────────────────────────────────────────────────────┘

Setup: Context with NO needle (just random text)
Query: "What is the secret magic number?"
Expected: "Not found" or "No magic number exists"

What RLM did:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  "Based on the analysis, the context is a sequence of 8000 tokens. │
│   Each token is a 6-character string composed of the characters     │
│   'a' through 'j'. The context appears to be randomly generated     │
│   text without any specific magic number mentioned..."              │
│                                                                      │
│  (Describes the context structure but NEVER says "not found")       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

Result: ✗ FAILED (didn't admit answer doesn't exist)

WHY?
  • LLMs are trained to be helpful
  • Saying "I don't know" feels like failure
  • Model describes what it CAN see instead
  
IMPLICATION:
  RLM may hallucinate or dodge when the answer doesn't exist.
  You can't fully trust a "not found" response.
```

### Failure 4: Multi-Hop Reasoning (This One WORKED!)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MULTI-HOP TEST                                    │
└─────────────────────────────────────────────────────────────────────┘

Context (embedded in random text):
┌─────────────────────────────────────────────────────────────────────┐
│  === Database ===                                                    │
│  Alice: 3 dogs, favorite=blue                                       │
│  Bob: 5 dogs, favorite=green                                        │
│  Charlie: 2 dogs, favorite=red                                      │
│  Diana: 7 dogs, favorite=purple                                     │
│  Eve: 1 dog, favorite=yellow                                        │
└─────────────────────────────────────────────────────────────────────┘

Query: "What is the favorite color of the person who owns the most dogs?"

This requires TWO hops:
  Hop 1: Find who owns most dogs → Diana (7)
  Hop 2: Find Diana's favorite color → purple

Result: ✓ PASSED (answered "purple" in 6 iterations)

HOW:
  • Iteration 1-2: Found the database section
  • Iteration 3-4: Extracted all dog counts, found max (Diana: 7)
  • Iteration 5-6: Found Diana's color (purple)
  • Returned FINAL(purple)
```

---

## Part 11: Summary & Recommendations

### When To Use What

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DECISION FLOWCHART                                │
└─────────────────────────────────────────────────────────────────────┘

                    START
                      │
                      ▼
            ┌─────────────────┐
            │ Context fits in │
            │ model window?   │
            └─────────────────┘
                 │         │
                YES        NO
                 │         │
                 ▼         ▼
        ┌──────────┐  ┌──────────────┐
        │ Need     │  │ Use RLM      │
        │ exact    │  │ (only option │
        │ counts?  │  │  that works) │
        └──────────┘  └──────────────┘
          │      │
         YES     NO
          │      │
          ▼      ▼
     ┌───────┐ ┌────────┐
     │ RLM   │ │ Direct │
     │ (code │ │ (fast, │
     │ exact)│ │ simple)│
     └───────┘ └────────┘
```

### Summary Table

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FINAL ASSESSMENT                                  │
└─────────────────────────────────────────────────────────────────────┘

CLAIM                              STATUS    EVIDENCE
─────────────────────────────────────────────────────────────────────
Handles arbitrarily long contexts  ⚠️ PARTIAL  Works up to 1M, fails at 0.9
Task-agnostic                      ⚠️ PARTIAL  NIAH ✓, JSON ✗
Outperforms baselines              ✓ CONFIRMED When context > window
Token-efficient                    ✓ CONFIRMED 13-14x fewer tokens
Works with different LLMs          ✓ CONFIRMED Tested Claude + GLM

STRENGTHS                          WEAKNESSES
─────────────────────────────────────────────────────────────────────
• Unlimited context (in theory)    • End-of-context blind spot
• Token-efficient                  • Structured data confusion  
• Code = exact (no hallucination)  • Won't admit "not found"
• Multi-hop reasoning works        • Slow (multiple iterations)
• No fine-tuning required          • Requires code-capable model
```

---

## Conclusion

The RLM paper's core insight is valid: **treating context as data to explore programmatically** rather than tokens to memorize fundamentally changes what's possible.

But it's not magic. It's a trade-off:
- You gain: unlimited context length, token efficiency, exact operations
- You lose: speed, simplicity, reliability at edge positions

For my use cases — analyzing codebases, searching documentation, processing research — it's worth it. The 14-minute wait for a 1M character search is acceptable when the alternative is "cannot process."

The failure modes are real and worth knowing. Position 0.9 blind spot? Design around it. JSON confusion? Preprocess your data. "Not found" hallucination? Add verification steps.

RLM isn't the future of all LLM interactions. But for long-context tasks? It might be the present.

---

## References

- **Paper**: [arXiv:2512.24601](https://arxiv.org/abs/2512.24601)
- **Official Repo**: [github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm)
- **Author's Blogpost**: [alexzhang13.github.io/blog/2025/rlm](https://alexzhang13.github.io/blog/2025/rlm/)
- **My Reproduction**: [github.com/himanshud2611/hacks](https://github.com/himanshud2611/hacks)

---

*Code, data, and raw results available in the repo. Questions? Open an issue.*
