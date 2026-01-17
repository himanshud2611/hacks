"""Comprehensive RLM Experiment Runner - Tests various hypotheses and edge cases"""
import os, sys, time, json, random
from datetime import datetime

sys.path.insert(0, '/workspace/tools/python-packages')
sys.path.insert(0, '/workspace/himanshud-hacks/rlm-reproduction')

from src.benchmarks.niah import generate_niah_sample
from src.core.anannas_client import AnannasClient
from src.core.rlm import RLM
from src.baselines.direct import DirectBaseline

MODEL = "zai-org/glm-4.7"
API_KEY = os.environ.get("ANANNAS_API_KEY")

def run_test(context, query, expected, method="rlm", max_iters=10):
    client = AnannasClient(model=MODEL, api_key=API_KEY)
    start = time.time()
    if method == "rlm":
        rlm = RLM(root_client=client, verbose=False, max_iterations=max_iters)
        result = rlm.completion(context, query)
        answer, iters = result.answer, len(result.iterations)
        tokens = result.root_usage.input_tokens
    else:
        direct = DirectBaseline(client=client, max_context_chars=200000, verbose=False)
        result = direct.completion(context, query)
        answer, iters, tokens = result.answer, 1, result.usage.input_tokens
    return {"answer": answer[:200], "expected": expected, "correct": expected.lower() in answer.lower(),
            "time": time.time()-start, "iterations": iters, "tokens": tokens}

def exp_a_position():
    """Test needle at different positions in 300K context"""
    print("\n" + "="*60 + "\nEXP A: Position Sweep (300K)\n" + "="*60)
    results = []
    for pos in [0.1, 0.3, 0.5, 0.7, 0.9]:
        print(f"[Pos {pos}] ", end="", flush=True)
        sample = generate_niah_sample(300000, pos, "number", int(pos*1000))
        r = run_test(sample.context, sample.query, sample.answer)
        r["position"] = pos
        results.append(r)
        print(f"{'✓' if r['correct'] else '✗'} {r['time']:.0f}s {r['iterations']}i")
    return results

def exp_b_structured():
    """Test JSON/CSV vs random words"""
    print("\n" + "="*60 + "\nEXP B: Structured Data (100K)\n" + "="*60)
    needle = "42857"
    query = "What is the secret magic number?"
    results = []
    
    # Random words
    print("[Random] ", end="", flush=True)
    sample = generate_niah_sample(100000, 0.5, "number", 42)
    r = run_test(sample.context, sample.query, sample.answer)
    r["format"] = "random"
    results.append(r)
    print(f"{'✓' if r['correct'] else '✗'} {r['time']:.0f}s")
    
    # JSON
    print("[JSON] ", end="", flush=True)
    random.seed(42)
    records = []
    for i in range(1000):
        if i == 500:
            records.append({"id": i, "type": "secret", "magic_number": needle})
        else:
            records.append({"id": i, "value": random.randint(1,999), "data": "x"*80})
    json_ctx = json.dumps(records, indent=2)
    r = run_test(json_ctx, query, needle)
    r["format"] = "json"
    results.append(r)
    print(f"{'✓' if r['correct'] else '✗'} {r['time']:.0f}s")
    
    return results

def exp_c_multi_needle():
    """Find multiple facts"""
    print("\n" + "="*60 + "\nEXP C: Multi-Needle (100K)\n" + "="*60)
    random.seed(123)
    words = [''.join(random.choices('abcdefghij', k=6)) for _ in range(15000)]
    text = ' '.join(words)
    # Insert 3 needles
    needles = {"secret_code": ("ALPHA7", 0.2), "magic_number": ("98765", 0.5), "password": ("hunter2", 0.8)}
    for name, (val, pos) in needles.items():
        p = int(len(text)*pos)
        text = text[:p] + f" [FACT: The {name} is {val}.] " + text[p:]
    
    results = []
    for name, (val, pos) in needles.items():
        print(f"[{name}] ", end="", flush=True)
        q = f"What is the {name.replace('_',' ')}?"
        r = run_test(text[:100000], q, val)
        r["needle"] = name
        results.append(r)
        print(f"{'✓' if r['correct'] else '✗'} {r['time']:.0f}s")
    return results

def exp_d_adversarial():
    """Test with decoy needles"""
    print("\n" + "="*60 + "\nEXP D: Adversarial Decoys (100K)\n" + "="*60)
    random.seed(456)
    words = [''.join(random.choices('abcdefghij', k=6)) for _ in range(15000)]
    text = ' '.join(words)
    # Decoys
    for i, decoy in enumerate(["11111", "22222", "33333"]):
        p = int(len(text)*(0.2+i*0.15))
        text = text[:p] + f" [Note: Some say magic number is {decoy}, but WRONG.] " + text[p:]
    # Real
    p = int(len(text)*0.75)
    text = text[:p] + " [IMPORTANT: TRUE magic number is 77777. This is CORRECT.] " + text[p:]
    
    print("[Decoys] ", end="", flush=True)
    r = run_test(text[:100000], "What is the TRUE magic number?", "77777", max_iters=12)
    print(f"{'✓' if r['correct'] else '✗'} {r['time']:.0f}s")
    print(f"Answer: {r['answer'][:100]}")
    return r

def exp_e_no_answer():
    """Query something that doesn't exist"""
    print("\n" + "="*60 + "\nEXP E: No Answer Exists (50K)\n" + "="*60)
    random.seed(789)
    text = ' '.join([''.join(random.choices('abcdefghij', k=6)) for _ in range(8000)])
    
    print("[No needle] ", end="", flush=True)
    client = AnannasClient(model=MODEL, api_key=API_KEY)
    rlm = RLM(root_client=client, verbose=False, max_iterations=8)
    start = time.time()
    result = rlm.completion(text, "What is the secret magic number?")
    
    neg_phrases = ["not found", "no magic", "doesn't exist", "not mentioned", "cannot find", "none"]
    correct = any(p in result.answer.lower() for p in neg_phrases)
    print(f"{'✓' if correct else '✗'} {time.time()-start:.0f}s")
    print(f"Answer: {result.answer[:150]}")
    return {"correct": correct, "answer": result.answer[:200]}

def exp_f_multi_hop():
    """Multi-hop reasoning"""
    print("\n" + "="*60 + "\nEXP F: Multi-Hop Reasoning\n" + "="*60)
    facts = """
=== Database ===
Alice: 3 dogs, favorite=blue
Bob: 5 dogs, favorite=green  
Charlie: 2 dogs, favorite=red
Diana: 7 dogs, favorite=purple
Eve: 1 dog, favorite=yellow
"""
    random.seed(999)
    padding = ' '.join([''.join(random.choices('abcdefghij', k=6)) for _ in range(12000)])
    context = padding[:40000] + facts + padding[40000:]
    
    queries = [("Who owns the most dogs?", "diana", "1-hop"),
               ("What is Diana's favorite color?", "purple", "1-hop"),
               ("What is the favorite color of the person with most dogs?", "purple", "2-hop")]
    results = []
    for q, exp, hops in queries:
        print(f"[{hops}] ", end="", flush=True)
        r = run_test(context, q, exp)
        r["hops"] = hops
        results.append(r)
        print(f"{'✓' if r['correct'] else '✗'} {r['time']:.0f}s {r['iterations']}i")
    return results

if __name__ == "__main__":
    if not API_KEY:
        print("Set ANANNAS_API_KEY"); exit(1)
    
    exps = sys.argv[1:] if len(sys.argv) > 1 else ['a','b','c','d','e','f']
    results = {}
    
    if 'a' in exps: results['A'] = exp_a_position()
    if 'b' in exps: results['B'] = exp_b_structured()
    if 'c' in exps: results['C'] = exp_c_multi_needle()
    if 'd' in exps: results['D'] = exp_d_adversarial()
    if 'e' in exps: results['E'] = exp_e_no_answer()
    if 'f' in exps: results['F'] = exp_f_multi_hop()
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"/workspace/himanshud-hacks/rlm-reproduction/results/exp_{ts}.json"
    with open(path, 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {path}")
