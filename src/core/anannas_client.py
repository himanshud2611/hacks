"""
Anannas API Client (OpenAI-compatible)
https://api.anannas.ai/v1
"""

import os
import time
from typing import List, Dict, Optional

from .llm_client import LLMClient, UsageStats


class AnannasClient(LLMClient):
    """
    Anannas API client - OpenAI compatible.
    
    Models use format: provider/model-name
    e.g., anthropic/claude-3-sonnet, openai/gpt-4o
    """
    
    def __init__(
        self, 
        model: str = "anthropic/claude-3.5-sonnet",
        api_key: Optional[str] = None,
        base_url: str = "https://api.anannas.ai/v1"
    ):
        super().__init__(model)
        self.api_key = api_key or os.environ.get("ANANNAS_API_KEY")
        self.base_url = base_url
        
        if not self.api_key:
            raise ValueError("ANANNAS_API_KEY not set")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        except ImportError:
            raise ImportError("openai package required: pip install openai")
    
    def completion(self, messages: List[Dict[str, str]], max_tokens: int = 8192, **kwargs) -> str:
        """Generate completion via Anannas API."""
        start_time = time.time()
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
        )
        
        result = response.choices[0].message.content
        
        # Track usage
        elapsed = time.time() - start_time
        input_tokens = getattr(response.usage, 'prompt_tokens', 0)
        output_tokens = getattr(response.usage, 'completion_tokens', 0)
        self.usage.add(input_tokens, output_tokens, elapsed)
        
        return result
