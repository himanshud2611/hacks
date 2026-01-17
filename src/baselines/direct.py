"""
Direct Prompting Baseline

Simply puts the entire context in the prompt.
Limited by context window size.
"""

import time
from dataclasses import dataclass
from typing import Any, List, Optional

from ..core.llm_client import LLMClient, UsageStats


@dataclass
class DirectResult:
    """Result from direct prompting."""
    answer: str
    total_time: float
    usage: UsageStats
    truncated: bool = False
    success: bool = True


class DirectBaseline:
    """
    Direct prompting baseline.
    
    Simply puts the context directly in the prompt.
    Truncates if context exceeds max_context_chars.
    """
    
    def __init__(
        self,
        client: LLMClient,
        max_context_chars: int = 100000,  # ~25K tokens
        verbose: bool = False,
    ):
        self.client = client
        self.max_context_chars = max_context_chars
        self.verbose = verbose
    
    def _log(self, msg: str):
        if self.verbose:
            print(f"[Direct] {msg}")
    
    def completion(
        self,
        context: Any,
        query: Optional[str] = None,
    ) -> DirectResult:
        """
        Direct completion with full context in prompt.
        
        Args:
            context: The input context
            query: The question to answer
            
        Returns:
            DirectResult
        """
        start_time = time.time()
        self.client.reset_usage()
        
        # Convert context to string
        if isinstance(context, str):
            context_str = context
        elif isinstance(context, list):
            context_str = "\n\n".join(str(c) for c in context)
        elif isinstance(context, dict):
            import json
            context_str = json.dumps(context, indent=2)
        else:
            context_str = str(context)
        
        # Truncate if needed
        truncated = False
        if len(context_str) > self.max_context_chars:
            self._log(f"Truncating context from {len(context_str)} to {self.max_context_chars} chars")
            context_str = context_str[:self.max_context_chars]
            truncated = True
        
        # Build prompt
        prompt = f"""Answer the following query based on the provided context.

Context:
{context_str}

Query: {query or "Please analyze the context and provide a summary."}

Answer:"""
        
        messages = [{"role": "user", "content": prompt}]
        
        self._log(f"Sending prompt ({len(prompt)} chars)")
        response = self.client.completion(messages)
        
        return DirectResult(
            answer=response,
            total_time=time.time() - start_time,
            usage=self.client.get_usage(),
            truncated=truncated,
            success=True,
        )
