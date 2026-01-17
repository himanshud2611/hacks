"""
Recursive Language Model (RLM) - Core Implementation

This implements the RLM inference strategy from the paper:
"Recursive Language Models" (arXiv:2512.24601)

Key idea: Offload context to a REPL environment, let the model
programmatically explore it via code and recursive sub-LLM calls.
"""

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .llm_client import LLMClient, UsageStats
from .repl import REPLEnvironment, REPLResult


# System prompt for RLM
RLM_SYSTEM_PROMPT = """You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs.

The REPL environment is initialized with:
1. A `context` variable containing the input data. ALWAYS check this first.
2. A `llm_query(prompt)` function to query a sub-LLM (~500K char capacity).
3. A `llm_query_batched(prompts)` function for concurrent sub-LLM queries.
4. `print()` statements to view outputs.

IMPORTANT STRATEGIES:
- First examine the context structure: type, length, format
- For long contexts: chunk and query sub-LLMs on each chunk
- Use llm_query_batched for multiple independent queries (faster)
- Build up your answer incrementally using variables as buffers

To execute Python code, wrap it in triple backticks with 'repl':
```repl
# Example: explore context
print(type(context))
print(len(context) if hasattr(context, '__len__') else 'no length')
```

Example - chunking and querying:
```repl
query = "What is the main topic?"
chunk_size = len(context) // 5
answers = []
for i in range(5):
    start, end = i * chunk_size, (i + 1) * chunk_size if i < 4 else len(context)
    chunk = context[start:end]
    answer = llm_query(f"Answer: {query}\\nContext: {chunk}")
    answers.append(answer)
final = llm_query(f"Combine these answers: {answers}")
print(final)
```

FINAL ANSWER:
When done, provide your answer using ONE of:
1. FINAL(your answer here) - direct answer
2. FINAL_VAR(variable_name) - return a variable's value

DO NOT provide a final answer until you have examined the context.
Think step-by-step and execute code immediately."""


@dataclass
class RLMIteration:
    """Record of a single RLM iteration."""
    iteration: int
    response: str
    code_blocks: List[str] = field(default_factory=list)
    code_results: List[REPLResult] = field(default_factory=list)
    has_final_answer: bool = False
    final_answer: Optional[str] = None
    iteration_time: float = 0.0


@dataclass 
class RLMResult:
    """Final result from RLM completion."""
    answer: str
    iterations: List[RLMIteration]
    total_time: float
    root_usage: UsageStats
    sub_usage: UsageStats
    success: bool = True
    
    def __str__(self):
        return f"RLMResult(answer={self.answer[:100]}..., iterations={len(self.iterations)}, time={self.total_time:.2f}s)"


class RLM:
    """
    Recursive Language Model.
    
    Replaces standard LLM completion with an RLM completion that can
    handle arbitrarily long contexts through recursive decomposition.
    """
    
    def __init__(
        self,
        root_client: LLMClient,
        sub_client: Optional[LLMClient] = None,
        max_iterations: int = 20,
        max_output_chars: int = 3000,
        verbose: bool = False,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize RLM.
        
        Args:
            root_client: LLM client for the root model
            sub_client: LLM client for sub-LLM queries (defaults to root_client)
            max_iterations: Maximum REPL iterations before forcing answer
            max_output_chars: Max chars to show from REPL output
            verbose: Print progress
            system_prompt: Custom system prompt (uses default if None)
        """
        self.root_client = root_client
        self.sub_client = sub_client or root_client
        self.max_iterations = max_iterations
        self.max_output_chars = max_output_chars
        self.verbose = verbose
        self.system_prompt = system_prompt or RLM_SYSTEM_PROMPT
        
        # Track usage
        self.sub_usage = UsageStats()
    
    def _log(self, msg: str):
        """Print if verbose."""
        if self.verbose:
            print(f"[RLM] {msg}")
    
    def _extract_code_blocks(self, response: str) -> List[str]:
        """Extract ```repl code blocks from response."""
        pattern = r'```repl\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        return matches
    
    def _extract_final_answer(self, response: str, repl: REPLEnvironment) -> Optional[str]:
        """Extract FINAL() or FINAL_VAR() from response."""
        # Check for FINAL(answer)
        final_match = re.search(r'FINAL\((.*?)\)', response, re.DOTALL)
        if final_match:
            return final_match.group(1).strip()
        
        # Check for FINAL_VAR(variable)
        var_match = re.search(r'FINAL_VAR\(([^)]+)\)', response)
        if var_match:
            var_name = var_match.group(1).strip().strip('"').strip("'")
            value = repl.get_variable(var_name)
            if value is not None:
                return str(value)
            return f"[Variable '{var_name}' not found]"
        
        return None
    
    def _llm_query(self, prompt: str) -> str:
        """Sub-LLM query function for REPL environment."""
        start_time = time.time()
        
        messages = [{"role": "user", "content": prompt}]
        response = self.sub_client.completion(messages)
        
        elapsed = time.time() - start_time
        self._log(f"Sub-LLM query completed in {elapsed:.2f}s")
        
        return response
    
    def _llm_query_batched(self, prompts: List[str]) -> List[str]:
        """Batched sub-LLM queries (sequential for now, could be parallel)."""
        results = []
        for i, prompt in enumerate(prompts):
            self._log(f"Batched query {i+1}/{len(prompts)}")
            results.append(self._llm_query(prompt))
        return results
    
    def _build_messages(
        self,
        query: Optional[str],
        iteration: int,
        history: List[Dict[str, str]],
        context_info: str,
    ) -> List[Dict[str, str]]:
        """Build message list for root LLM."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "assistant", "content": f"Context info: {context_info}"},
        ]
        
        # Add history
        messages.extend(history)
        
        # Add user prompt
        if iteration == 0:
            user_content = "You have not examined the context yet. First, explore the context variable, then work toward answering the query.\n\n"
        else:
            user_content = "Continue working toward the answer. "
        
        if query:
            user_content += f"Query to answer: {query}\n\n"
        user_content += "Your next action:"
        
        messages.append({"role": "user", "content": user_content})
        
        return messages
    
    def _get_context_info(self, context: Any) -> str:
        """Get info about context for the model."""
        if isinstance(context, str):
            return f"String with {len(context)} characters"
        elif isinstance(context, list):
            return f"List with {len(context)} items"
        elif isinstance(context, dict):
            return f"Dict with keys: {list(context.keys())[:10]}"
        else:
            return f"Type: {type(context).__name__}"
    
    def completion(
        self,
        context: Any,
        query: Optional[str] = None,
    ) -> RLMResult:
        """
        RLM completion - handles arbitrarily long contexts.
        
        Args:
            context: The input context (string, list, dict, etc.)
            query: The question to answer about the context
            
        Returns:
            RLMResult with the answer and metadata
        """
        start_time = time.time()
        self._log(f"Starting RLM completion")
        
        # Reset usage tracking
        self.root_client.reset_usage()
        self.sub_client.reset_usage()
        
        # Create REPL environment
        repl = REPLEnvironment(
            context=context,
            llm_query_fn=self._llm_query,
            llm_query_batched_fn=self._llm_query_batched,
            max_output_chars=self.max_output_chars,
        )
        
        context_info = self._get_context_info(context)
        self._log(f"Context: {context_info}")
        
        iterations = []
        history = []
        
        try:
            for i in range(self.max_iterations):
                self._log(f"Iteration {i+1}/{self.max_iterations}")
                iter_start = time.time()
                
                # Build messages and get response
                messages = self._build_messages(query, i, history, context_info)
                response = self.root_client.completion(messages)
                
                # Extract and execute code blocks
                code_blocks = self._extract_code_blocks(response)
                code_results = []
                
                for j, code in enumerate(code_blocks):
                    self._log(f"Executing code block {j+1}")
                    result = repl.execute(code)
                    code_results.append(result)
                    self._log(f"Output: {result.truncated(500)}")
                
                # Check for final answer
                final_answer = self._extract_final_answer(response, repl)
                
                # Record iteration
                iteration = RLMIteration(
                    iteration=i,
                    response=response,
                    code_blocks=code_blocks,
                    code_results=code_results,
                    has_final_answer=final_answer is not None,
                    final_answer=final_answer,
                    iteration_time=time.time() - iter_start,
                )
                iterations.append(iteration)
                
                # Return if we have final answer
                if final_answer:
                    self._log(f"Final answer found at iteration {i+1}")
                    return RLMResult(
                        answer=final_answer,
                        iterations=iterations,
                        total_time=time.time() - start_time,
                        root_usage=self.root_client.get_usage(),
                        sub_usage=self.sub_client.get_usage(),
                        success=True,
                    )
                
                # Update history for next iteration
                history.append({"role": "assistant", "content": response})
                
                # Add code results to history
                if code_results:
                    results_text = "\n".join([
                        f"Code block {j+1} output:\n{r.truncated(1000)}"
                        for j, r in enumerate(code_results)
                    ])
                    history.append({"role": "user", "content": f"REPL results:\n{results_text}"})
            
            # Max iterations reached - force final answer
            self._log("Max iterations reached, forcing final answer")
            messages = history + [
                {"role": "user", "content": "You must provide a FINAL() answer now based on what you've learned."}
            ]
            response = self.root_client.completion(messages)
            final_answer = self._extract_final_answer(response, repl)
            
            if not final_answer:
                final_answer = response  # Use raw response as fallback
            
            return RLMResult(
                answer=final_answer,
                iterations=iterations,
                total_time=time.time() - start_time,
                root_usage=self.root_client.get_usage(),
                sub_usage=self.sub_client.get_usage(),
                success=True,
            )
            
        finally:
            repl.cleanup()


# Convenience function
def rlm_completion(
    context: Any,
    query: Optional[str] = None,
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-20250514",
    verbose: bool = False,
    **kwargs
) -> RLMResult:
    """
    Convenience function for RLM completion.
    
    Args:
        context: Input context
        query: Question to answer
        provider: LLM provider ("anthropic" or "openai")
        model: Model name
        verbose: Print progress
        
    Returns:
        RLMResult
    """
    from .llm_client import get_client
    
    client = get_client(provider, model)
    rlm = RLM(root_client=client, verbose=verbose, **kwargs)
    return rlm.completion(context, query)
