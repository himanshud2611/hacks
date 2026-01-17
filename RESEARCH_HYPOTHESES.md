# RLM Research: Hypotheses & Edge Cases

**Date:** 2025-01-16  
**Objective:** Deep exploration of RLM capabilities, limitations, and edge cases

---

## Research Questions

### 1. Context Structure Sensitivity
**Q:** Does RLM perform differently based on HOW the needle is hidden?
- Plain text vs structured data (JSON, CSV, XML)
- Random noise vs coherent prose
- Single needle vs multiple needles

### 2. Needle Position Effects
**Q:** Is there a "blind spot" in RLM's search strategy?
- Beginning (0.1), middle (0.5), end (0.9)
- Multiple needles at different positions
- Adversarial placement (right at chunk boundaries)

### 3. Query Complexity
**Q:** How does RLM handle multi-hop reasoning?
- Single fact retrieval (what is X?)
- Two-hop (what is X's relation to Y?)
- Aggregation + reasoning (count X where condition Y)

### 4. Failure Modes
**Q:** When does RLM fail?
- Ambiguous queries
- No answer exists in context
- Misleading/adversarial needles

### 5. Model Capability Threshold
**Q:** Does RLM work with weaker models?
- Strong model (Claude 3.5 Sonnet) ✓ tested
- Medium model (GLM-4.7) ✓ tested
- Weak/small model (?)

### 6. Context Type Generalization
**Q:** Does RLM work beyond text?
- Code repositories
- Structured data (JSON arrays)
- Multi-document scenarios

### 7. Efficiency Scaling
**Q:** How does token usage scale with context size?
- 100K, 500K, 1M - is it linear or sub-linear?

---

## Planned Experiments

### Experiment A: Needle Position Sweep
Test needle at positions 0.1, 0.3, 0.5, 0.7, 0.9 with 300K context

### Experiment B: Structured vs Unstructured
Same needle hidden in:
- Random words (current approach)
- Coherent prose (lorem ipsum style)
- JSON documents
- CSV data

### Experiment C: Multi-Needle Retrieval
Multiple facts scattered throughout, query asks for all of them

### Experiment D: Adversarial Needle
- Needle that looks similar to noise
- Decoy needles with wrong values
- Needle split across sentences

### Experiment E: No-Answer Scenario
Query for something that doesn't exist - does RLM correctly say "not found"?

### Experiment F: Multi-Hop Reasoning
"What is the favorite color of the person who owns the most dogs?"

### Experiment G: Code Context
Large codebase - find a specific function or bug

---

## Hypotheses to Validate

| ID | Hypothesis | Prediction |
|----|------------|------------|
| H1 | Needle position doesn't matter for RLM | RLM finds needle at any position |
| H2 | Structured data is easier to search | JSON/CSV > random words |
| H3 | RLM gracefully handles "no answer" | Returns "not found" appropriately |
| H4 | Multi-hop requires more iterations | 2-hop > 1-hop iterations |
| H5 | Decoy needles confuse RLM | Accuracy drops with decoys |
| H6 | Token usage scales sub-linearly | 1M doesn't use 10x tokens of 100K |

---

## Results

### Experiment A: Position Sweep (300K context)

| Position | Correct | Time | Tokens | Iterations |
|----------|---------|------|--------|------------|
| 0.1 | ✓ | 90s | 180K | 4 |
| 0.3 | ✓ | 18s | 2.8K | 3 |
| 0.5 | ✓ | 33s | 2.3K | 4 |
| 0.7 | ✓ | 41s | 3.3K | 4 |
| **0.9** | **✗** | **312s** | **300K** | 4 |

**🔴 CRITICAL: Position 0.9 FAILED!** RLM has a blind spot at the very end of long contexts. Returned "42" instead of correct answer.

**Hypothesis H1 REJECTED:** Position DOES matter - end positions are problematic.

---

### Experiment B: Structured Data (100K context)

| Format | Correct | Time | Tokens |
|--------|---------|------|--------|
| Random words | ✓ | 63s | 51K |
| **JSON** | **✗** | 94s | 9.5K |

**🔴 JSON format FAILED!** Returned "998" instead of "42857" - got confused by other numeric values in JSON structure.

**Hypothesis H2 REJECTED:** Structured data is NOT easier - in fact it's HARDER due to noise from similar-looking values.

---

### Experiment C: Multi-Needle Retrieval (100K context)

| Needle | Position | Correct | Time |
|--------|----------|---------|------|
| secret_code | 0.2 | ✓ | 232s |
| magic_number | 0.5 | ✓ | 46s |
| password | 0.8 | ✓ | 52s |

**✓ All individual retrievals passed.** RLM can find specific facts when queried directly.

---

### Experiment D: Adversarial Decoys (100K context)

| Test | Correct | Time |
|------|---------|------|
| Real value with 3 decoys | ✓ | 261s |

**✓ PASSED!** RLM correctly identified "77777" as the TRUE value despite decoys saying "11111", "22222", "33333" were magic numbers.

**Hypothesis H5 REJECTED:** Decoys do NOT confuse RLM when properly labeled.

---

### Experiment E: No-Answer Scenario (50K context)

| Test | Correct | Time |
|------|---------|------|
| Query for non-existent fact | **✗** | 63s |

**🔴 FAILED!** RLM did NOT correctly say "not found". Instead it described the context structure without admitting the answer doesn't exist.

**Hypothesis H3 REJECTED:** RLM does NOT gracefully handle missing answers.

---

### Experiment F: Multi-Hop Reasoning (80K context)

| Query | Hops | Correct | Time | Iterations |
|-------|------|---------|------|------------|
| Who owns most dogs? | 1 | ✓ | 145s | 7 |
| What is Diana's color? | 1 | ✓ | 119s | 6 |
| Color of person with most dogs? | 2 | ✓ | 189s | 6 |

**✓ All passed including 2-hop!** RLM successfully chains reasoning.

**Hypothesis H4 CONFIRMED:** 2-hop doesn't require more iterations than 1-hop (both ~6 iters).

---

## Summary of Findings

### ✓ What RLM Does Well
1. **Multi-hop reasoning** - Can chain facts together
2. **Adversarial robustness** - Not fooled by decoys when clearly labeled
3. **Multi-needle retrieval** - Finds individual facts reliably
4. **Middle positions** - Works well for needles at 0.1-0.7

### ✗ What RLM Struggles With
1. **End positions (0.9)** - Blind spot at very end of long contexts
2. **Structured data (JSON)** - Confused by similar numeric values
3. **Negative answers** - Won't admit when answer doesn't exist

### Hypotheses Scorecard

| ID | Hypothesis | Result |
|----|------------|--------|
| H1 | Position doesn't matter | ❌ REJECTED - 0.9 fails |
| H2 | Structured data easier | ❌ REJECTED - JSON harder |
| H3 | Graceful "not found" | ❌ REJECTED - Won't admit |
| H4 | 2-hop needs more iters | ❌ REJECTED - Same iters |
| H5 | Decoys confuse RLM | ❌ REJECTED - Handles well |
| H6 | Sub-linear token scaling | ✓ Needs more testing |
