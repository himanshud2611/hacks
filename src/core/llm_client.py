"""
LLM Client interfaces for RLM.
Supports multiple providers: Anthropic, OpenAI, etc.
"""

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class UsageStats:
    """Track token usage and costs."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_calls: int = 0
    total_time: float = 0.0
    
    def add(self, input_tokens: int, output_tokens: int, time_taken: float):
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_calls += 1
        self.total_time += time_taken
    
    def __str__(self):
        return f"UsageStats(calls={self.total_calls}, input={self.input_tokens}, output={self.output_tokens}, time={self.total_time:.2f}s)"


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, model: str):
        self.model = model
        self.usage = UsageStats()
    
    @abstractmethod
    def completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a completion from the model."""
        pass
    
    def get_usage(self) -> UsageStats:
        return self.usage
    
    def reset_usage(self):
        self.usage = UsageStats()


class AnthropicClient(LLMClient):
    """Anthropic Claude client."""
    
    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: Optional[str] = None):
        super().__init__(model)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")
    
    def completion(self, messages: List[Dict[str, str]], max_tokens: int = 8192, **kwargs) -> str:
        """Generate completion using Claude."""
        start_time = time.time()
        
        # Separate system message from other messages
        system_msg = None
        chat_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                chat_messages.append(msg)
        
        # Build request
        request_kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": chat_messages,
        }
        
        if system_msg:
            request_kwargs["system"] = system_msg
        
        # Make request
        response = self.client.messages.create(**request_kwargs)
        
        # Extract response text
        result = response.content[0].text
        
        # Track usage
        elapsed = time.time() - start_time
        self.usage.add(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            time_taken=elapsed
        )
        
        return result


class OpenAIClient(LLMClient):
    """OpenAI GPT client."""
    
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        super().__init__(model)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package required: pip install openai")
    
    def completion(self, messages: List[Dict[str, str]], max_tokens: int = 8192, **kwargs) -> str:
        """Generate completion using GPT."""
        start_time = time.time()
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
        )
        
        result = response.choices[0].message.content
        
        # Track usage
        elapsed = time.time() - start_time
        self.usage.add(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            time_taken=elapsed
        )
        
        return result


def get_client(provider: str, model: str, **kwargs) -> LLMClient:
    """Factory function to get an LLM client."""
    if provider == "anthropic":
        return AnthropicClient(model=model, **kwargs)
    elif provider == "openai":
        return OpenAIClient(model=model, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
