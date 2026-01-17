"""
RAG (Retrieval-Augmented Generation) Baseline

Uses BM25 retrieval to find relevant chunks before prompting.
"""

import math
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..core.llm_client import LLMClient, UsageStats


def tokenize(text: str) -> List[str]:
    """Simple tokenization."""
    import re
    return re.findall(r'\b\w+\b', text.lower())


class BM25:
    """Simple BM25 implementation for retrieval."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs: Dict[str, int] = {}
        self.doc_lens: List[int] = []
        self.avg_doc_len: float = 0
        self.docs: List[List[str]] = []
        self.n_docs: int = 0
    
    def fit(self, documents: List[str]):
        """Index documents."""
        self.docs = [tokenize(doc) for doc in documents]
        self.n_docs = len(self.docs)
        self.doc_lens = [len(doc) for doc in self.docs]
        self.avg_doc_len = sum(self.doc_lens) / self.n_docs if self.n_docs > 0 else 0
        
        # Calculate document frequencies
        for doc in self.docs:
            seen = set()
            for token in doc:
                if token not in seen:
                    self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
                    seen.add(token)
    
    def _score(self, query_tokens: List[str], doc_idx: int) -> float:
        """Calculate BM25 score for a document."""
        doc = self.docs[doc_idx]
        doc_len = self.doc_lens[doc_idx]
        doc_counter = Counter(doc)
        
        score = 0.0
        for token in query_tokens:
            if token not in doc_counter:
                continue
            
            tf = doc_counter[token]
            df = self.doc_freqs.get(token, 0)
            
            if df == 0:
                continue
            
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
            tf_norm = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len))
            
            score += idf * tf_norm
        
        return score
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Search for relevant documents."""
        query_tokens = tokenize(query)
        scores = [(i, self._score(query_tokens, i)) for i in range(self.n_docs)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


@dataclass
class RAGResult:
    """Result from RAG baseline."""
    answer: str
    total_time: float
    usage: UsageStats
    retrieved_indices: List[int]
    num_chunks_used: int
    success: bool = True


class RAGBaseline:
    """
    RAG (Retrieval-Augmented Generation) baseline.
    
    Uses BM25 to retrieve relevant chunks, then prompts with those chunks.
    """
    
    def __init__(
        self,
        client: LLMClient,
        top_k: int = 10,
        max_context_chars: int = 50000,
        verbose: bool = False,
    ):
        self.client = client
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.verbose = verbose
    
    def _log(self, msg: str):
        if self.verbose:
            print(f"[RAG] {msg}")
    
    def _chunk_text(self, text: str, chunk_size: int = 2000) -> List[str]:
        """Split text into chunks."""
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])
        return chunks
    
    def completion(
        self,
        context: Any,
        query: Optional[str] = None,
    ) -> RAGResult:
        """
        RAG completion with retrieval.
        
        Args:
            context: The input context (string or list of strings)
            query: The question to answer
            
        Returns:
            RAGResult
        """
        start_time = time.time()
        self.client.reset_usage()
        
        query = query or "Please analyze the context and provide a summary."
        
        # Prepare chunks
        if isinstance(context, str):
            chunks = self._chunk_text(context)
        elif isinstance(context, list):
            chunks = [str(c) for c in context]
        else:
            chunks = self._chunk_text(str(context))
        
        self._log(f"Indexing {len(chunks)} chunks")
        
        # Build BM25 index
        bm25 = BM25()
        bm25.fit(chunks)
        
        # Retrieve relevant chunks
        results = bm25.search(query, top_k=self.top_k)
        retrieved_indices = [idx for idx, score in results]
        
        self._log(f"Retrieved {len(retrieved_indices)} chunks")
        
        # Build context from retrieved chunks
        retrieved_chunks = []
        total_chars = 0
        
        for idx in retrieved_indices:
            chunk = chunks[idx]
            if total_chars + len(chunk) > self.max_context_chars:
                break
            retrieved_chunks.append(f"[Chunk {idx + 1}]\n{chunk}")
            total_chars += len(chunk)
        
        context_str = "\n\n".join(retrieved_chunks)
        
        # Build prompt
        prompt = f"""Answer the following query based on the relevant context passages retrieved.

Retrieved Context:
{context_str}

Query: {query}

Answer based only on the provided context:"""
        
        messages = [{"role": "user", "content": prompt}]
        
        self._log(f"Sending prompt ({len(prompt)} chars)")
        response = self.client.completion(messages)
        
        return RAGResult(
            answer=response,
            total_time=time.time() - start_time,
            usage=self.client.get_usage(),
            retrieved_indices=retrieved_indices,
            num_chunks_used=len(retrieved_chunks),
            success=True,
        )
