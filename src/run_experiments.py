"""
Experiment Runner for RLM Reproduction

Runs RLM and baselines on benchmarks, saves results.
"""

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.rlm import RLM, RLMResult
from src.core.llm_client import get_client, LLMClient
from src.baselines.direct import DirectBaseline
from src.baselines.rag import RAGBaseline
from src.baselines.chunked import ChunkedBaseline
from src.benchmarks.niah import NIAHBenchmark, NIAHSample
from src.benchmarks.aggregation import AggregationBenchmark, AggregationSample


@dataclass
class ExperimentResult:
    """Result of a single experiment."""
    method: str
    benchmark: str
    sample_id: int
    query: str
    expected: str
    predicted: str
    correct: bool
    time_taken: float
    input_tokens: int
    output_tokens: int
    context_length: int
    extra: Dict[str, Any]


class ExperimentRunner:
    """
    Runs experiments across methods and benchmarks.
    """
    
    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-20250514",
        results_dir: str = "results",
        verbose: bool = True,
    ):
        self.provider = provider
        self.model = model
        self.results_dir = results_dir
        self.verbose = verbose
        
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize client
        self.client = get_client(provider, model)
        
        # Initialize methods
        self.methods = {
            "rlm": self._run_rlm,
            "direct": self._run_direct,
            "rag": self._run_rag,
            "chunked": self._run_chunked,
        }
    
    def _log(self, msg: str):
        if self.verbose:
            print(f"[Experiment] {msg}")
    
    def _run_rlm(self, context: Any, query: str) -> Dict[str, Any]:
        """Run RLM method."""
        rlm = RLM(
            root_client=self.client,
            verbose=self.verbose,
            max_iterations=15,
        )
        result = rlm.completion(context, query)
        return {
            "answer": result.answer,
            "time": result.total_time,
            "input_tokens": result.root_usage.input_tokens + result.sub_usage.input_tokens,
            "output_tokens": result.root_usage.output_tokens + result.sub_usage.output_tokens,
            "iterations": len(result.iterations),
            "sub_calls": result.sub_usage.total_calls,
        }
    
    def _run_direct(self, context: Any, query: str) -> Dict[str, Any]:
        """Run direct prompting baseline."""
        baseline = DirectBaseline(
            client=self.client,
            verbose=self.verbose,
        )
        result = baseline.completion(context, query)
        return {
            "answer": result.answer,
            "time": result.total_time,
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "truncated": result.truncated,
        }
    
    def _run_rag(self, context: Any, query: str) -> Dict[str, Any]:
        """Run RAG baseline."""
        baseline = RAGBaseline(
            client=self.client,
            verbose=self.verbose,
        )
        result = baseline.completion(context, query)
        return {
            "answer": result.answer,
            "time": result.total_time,
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "chunks_used": result.num_chunks_used,
        }
    
    def _run_chunked(self, context: Any, query: str) -> Dict[str, Any]:
        """Run chunked baseline."""
        baseline = ChunkedBaseline(
            client=self.client,
            verbose=self.verbose,
        )
        result = baseline.completion(context, query)
        return {
            "answer": result.answer,
            "time": result.total_time,
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "num_chunks": result.num_chunks,
        }
    
    def run_niah_benchmark(
        self,
        methods: List[str] = None,
        context_lengths: List[int] = None,
        max_samples: int = None,
    ) -> List[ExperimentResult]:
        """Run NIAH benchmark."""
        methods = methods or ["rlm", "direct", "rag"]
        
        benchmark = NIAHBenchmark(
            context_lengths=context_lengths or [10000, 50000],
            needle_positions=[0.1, 0.5, 0.9],
            needle_types=["number", "fact"],
            samples_per_config=1,
        )
        
        samples = benchmark.generate_samples()
        if max_samples:
            samples = samples[:max_samples]
        
        self._log(f"Running NIAH benchmark with {len(samples)} samples")
        
        results = []
        for i, sample in enumerate(samples):
            self._log(f"\n=== Sample {i+1}/{len(samples)} ===")
            self._log(f"Context length: {sample.context_length}, Position: {sample.needle_position}")
            
            for method in methods:
                self._log(f"Running {method}...")
                
                try:
                    result = self.methods[method](sample.context, sample.query)
                    correct = benchmark.evaluate(result["answer"], sample.answer)
                    
                    exp_result = ExperimentResult(
                        method=method,
                        benchmark="niah",
                        sample_id=i,
                        query=sample.query,
                        expected=sample.answer,
                        predicted=result["answer"][:500],  # Truncate for storage
                        correct=correct,
                        time_taken=result["time"],
                        input_tokens=result.get("input_tokens", 0),
                        output_tokens=result.get("output_tokens", 0),
                        context_length=sample.context_length,
                        extra={
                            "needle_position": sample.needle_position,
                            "method_specific": {k: v for k, v in result.items() if k != "answer"},
                        }
                    )
                    results.append(exp_result)
                    
                    self._log(f"  Result: {'✓' if correct else '✗'} ({result['time']:.2f}s)")
                    
                except Exception as e:
                    self._log(f"  Error: {e}")
                    results.append(ExperimentResult(
                        method=method,
                        benchmark="niah",
                        sample_id=i,
                        query=sample.query,
                        expected=sample.answer,
                        predicted=f"ERROR: {e}",
                        correct=False,
                        time_taken=0,
                        input_tokens=0,
                        output_tokens=0,
                        context_length=sample.context_length,
                        extra={"error": str(e)}
                    ))
        
        return results
    
    def run_aggregation_benchmark(
        self,
        methods: List[str] = None,
        max_samples: int = None,
    ) -> List[ExperimentResult]:
        """Run aggregation benchmark."""
        methods = methods or ["rlm", "chunked", "rag"]
        
        benchmark = AggregationBenchmark(
            num_chunks_list=[20, 50],
            chunk_sizes=[1000],
            query_types=["counting", "comparison"],
            samples_per_config=1,
        )
        
        samples = benchmark.generate_samples()
        if max_samples:
            samples = samples[:max_samples]
        
        self._log(f"Running aggregation benchmark with {len(samples)} samples")
        
        results = []
        for i, sample in enumerate(samples):
            self._log(f"\n=== Sample {i+1}/{len(samples)} ===")
            self._log(f"Type: {sample.query_type}, Chunks: {sample.num_chunks}")
            
            # Join chunks for context
            context = "\n\n".join(sample.context)
            
            for method in methods:
                self._log(f"Running {method}...")
                
                try:
                    result = self.methods[method](context, sample.query)
                    correct = benchmark.evaluate(result["answer"], sample.answer)
                    
                    exp_result = ExperimentResult(
                        method=method,
                        benchmark="aggregation",
                        sample_id=i,
                        query=sample.query,
                        expected=sample.answer,
                        predicted=result["answer"][:500],
                        correct=correct,
                        time_taken=result["time"],
                        input_tokens=result.get("input_tokens", 0),
                        output_tokens=result.get("output_tokens", 0),
                        context_length=sample.context_length,
                        extra={
                            "query_type": sample.query_type,
                            "num_chunks": sample.num_chunks,
                            "method_specific": {k: v for k, v in result.items() if k != "answer"},
                        }
                    )
                    results.append(exp_result)
                    
                    self._log(f"  Result: {'✓' if correct else '✗'} ({result['time']:.2f}s)")
                    
                except Exception as e:
                    self._log(f"  Error: {e}")
                    results.append(ExperimentResult(
                        method=method,
                        benchmark="aggregation",
                        sample_id=i,
                        query=sample.query,
                        expected=sample.answer,
                        predicted=f"ERROR: {e}",
                        correct=False,
                        time_taken=0,
                        input_tokens=0,
                        output_tokens=0,
                        context_length=sample.context_length,
                        extra={"error": str(e)}
                    ))
        
        return results
    
    def save_results(self, results: List[ExperimentResult], filename: str):
        """Save results to JSON."""
        filepath = os.path.join(self.results_dir, filename)
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "provider": self.provider,
            "model": self.model,
            "results": [asdict(r) for r in results],
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self._log(f"Saved results to {filepath}")
    
    def print_summary(self, results: List[ExperimentResult]):
        """Print summary of results."""
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        
        # Group by method and benchmark
        from collections import defaultdict
        grouped = defaultdict(list)
        for r in results:
            grouped[(r.benchmark, r.method)].append(r)
        
        for (benchmark, method), method_results in sorted(grouped.items()):
            correct = sum(1 for r in method_results if r.correct)
            total = len(method_results)
            accuracy = correct / total * 100 if total > 0 else 0
            avg_time = sum(r.time_taken for r in method_results) / total if total > 0 else 0
            avg_tokens = sum(r.input_tokens + r.output_tokens for r in method_results) / total if total > 0 else 0
            
            print(f"\n{benchmark.upper()} - {method.upper()}:")
            print(f"  Accuracy: {correct}/{total} ({accuracy:.1f}%)")
            print(f"  Avg Time: {avg_time:.2f}s")
            print(f"  Avg Tokens: {avg_tokens:.0f}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RLM experiments")
    parser.add_argument("--benchmark", choices=["niah", "aggregation", "all"], default="all")
    parser.add_argument("--methods", nargs="+", default=["rlm", "direct", "rag"])
    parser.add_argument("--provider", default="anthropic")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", default=True)
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(
        provider=args.provider,
        model=args.model,
        verbose=args.verbose,
    )
    
    all_results = []
    
    if args.benchmark in ["niah", "all"]:
        results = runner.run_niah_benchmark(
            methods=args.methods,
            max_samples=args.max_samples,
        )
        all_results.extend(results)
    
    if args.benchmark in ["aggregation", "all"]:
        results = runner.run_aggregation_benchmark(
            methods=args.methods,
            max_samples=args.max_samples,
        )
        all_results.extend(results)
    
    # Save and summarize
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runner.save_results(all_results, f"experiment_{timestamp}.json")
    runner.print_summary(all_results)


if __name__ == "__main__":
    main()
