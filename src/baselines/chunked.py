"""
Chunked Baseline (Chunk + Aggregate)

Processes context in chunks, then aggregates answers.
"""

import time
from dataclasses import dataclass
from typing import Any, List, Optional

from ..core.llm_client import LLMClient, UsageStats


@dataclass
class ChunkedResult:
    """Result from chunked baseline."""
    answer: str
    total_time: float
    usage: UsageStats
    num_chunks: int
    chunk_answers: List[str]
    success: bool = True


class ChunkedBaseline:
    """
    Chunked baseline (Chunk + Aggregate).
    
    1. Split context into chunks
    2. Query LLM on each chunk
    3. Aggregate chunk answers with final LLM call
    """
    
    def __init__(
        self,
        client: LLMClient,
        chunk_size: int = 20000,
        max_chunks: int = 20,
        verbose: bool = False,
    ):
        self.client = client
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks
        self.verbose = verbose
    
    def _log(self, msg: str):
        if self.verbose:
            print(f"[Chunked] {msg}")
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunks.append(text[i:i + self.chunk_size])
        return chunks[:self.max_chunks]
    
    def completion(
        self,
        context: Any,
        query: Optional[str] = None,
    ) -> ChunkedResult:
        """
        Chunked completion.
        
        Args:
            context: The input context
            query: The question to answer
            
        Returns:
            ChunkedResult
        """
        start_time = time.time()
        self.client.reset_usage()
        
        query = query or "Please analyze the context and provide a summary."
        
        # Prepare chunks
        if isinstance(context, str):
            chunks = self._chunk_text(context)
        elif isinstance(context, list):
            chunks = [str(c) for c in context][:self.max_chunks]
        else:
            chunks = self._chunk_text(str(context))
        
        self._log(f"Processing {len(chunks)} chunks")
        
        # Process each chunk
        chunk_answers = []
        for i, chunk in enumerate(chunks):
            self._log(f"Processing chunk {i+1}/{len(chunks)}")
            
            prompt = f"""Answer the following query based on this chunk of context. If the information is not present in this chunk, say "Not found in this chunk."

Context chunk {i+1}/{len(chunks)}:
{chunk}

Query: {query}

Answer:"""
            
            messages = [{"role": "user", "content": prompt}]
            response = self.client.completion(messages)
            chunk_answers.append(f"Chunk {i+1}: {response}")
        
        # Aggregate answers
        self._log("Aggregating chunk answers")
        
        aggregation_prompt = f"""Based on the following answers from different chunks of a document, provide a final comprehensive answer to the query.

Query: {query}

Answers from chunks:
{chr(10).join(chunk_answers)}

Final aggregated answer:"""
        
        messages = [{"role": "user", "content": aggregation_prompt}]
        final_answer = self.client.completion(messages)
        
        return ChunkedResult(
            answer=final_answer,
            total_time=time.time() - start_time,
            usage=self.client.get_usage(),
            num_chunks=len(chunks),
            chunk_answers=chunk_answers,
            success=True,
        )
