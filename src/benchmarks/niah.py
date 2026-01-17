"""
Needle in a Haystack (NIAH) Benchmark

Tests ability to find specific information buried in long contexts.
Inspired by the S-NIAH benchmark from the RLM paper.
"""

import random
import string
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class NIAHSample:
    """A single NIAH test sample."""
    context: str
    query: str
    answer: str
    needle_position: float  # 0.0 = start, 1.0 = end
    context_length: int
    needle: str


def generate_random_words(n: int, seed: Optional[int] = None) -> List[str]:
    """Generate random words for haystack."""
    if seed:
        random.seed(seed)
    
    words = []
    for _ in range(n):
        word_len = random.randint(3, 10)
        word = ''.join(random.choices(string.ascii_lowercase, k=word_len))
        words.append(word)
    return words


def generate_niah_sample(
    target_length: int = 100000,
    needle_position: float = 0.5,
    needle_type: str = "number",
    seed: Optional[int] = None,
) -> NIAHSample:
    """
    Generate a NIAH test sample.
    
    Args:
        target_length: Target context length in characters
        needle_position: Where to place needle (0.0 = start, 1.0 = end)
        needle_type: Type of needle ("number", "fact", "code")
        seed: Random seed
        
    Returns:
        NIAHSample with context, query, and expected answer
    """
    if seed:
        random.seed(seed)
    
    # Generate needle based on type
    if needle_type == "number":
        magic_number = random.randint(10000, 99999)
        needle = f"The secret magic number is {magic_number}. Remember this."
        query = "What is the secret magic number mentioned in the context?"
        answer = str(magic_number)
    
    elif needle_type == "fact":
        facts = [
            ("The capital of Zyntoria is Chromaville.", "What is the capital of Zyntoria?", "Chromaville"),
            ("Dr. Elena Voss invented the quantum bridge in 2089.", "Who invented the quantum bridge?", "Dr. Elena Voss"),
            ("The Crimson Protocol requires exactly 7 authentication steps.", "How many authentication steps does the Crimson Protocol require?", "7"),
            ("Project Nightingale was launched on March 15th, 2076.", "When was Project Nightingale launched?", "March 15th, 2076"),
        ]
        needle, query, answer = random.choice(facts)
    
    elif needle_type == "code":
        password = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
        needle = f"# IMPORTANT: The system password is: {password}"
        query = "What is the system password mentioned in the code comments?"
        answer = password
    
    else:
        raise ValueError(f"Unknown needle_type: {needle_type}")
    
    # Calculate haystack size
    needle_len = len(needle)
    haystack_len = target_length - needle_len - 100  # Leave some margin
    
    # Generate haystack
    words = generate_random_words(haystack_len // 6, seed)  # ~6 chars per word avg
    haystack = ' '.join(words)
    
    # Truncate or extend to target length
    if len(haystack) > haystack_len:
        haystack = haystack[:haystack_len]
    
    # Insert needle at specified position
    insert_pos = int(len(haystack) * needle_position)
    
    # Find word boundary
    while insert_pos > 0 and haystack[insert_pos-1] != ' ':
        insert_pos -= 1
    
    context = haystack[:insert_pos] + "\n\n" + needle + "\n\n" + haystack[insert_pos:]
    
    return NIAHSample(
        context=context,
        query=query,
        answer=answer,
        needle_position=needle_position,
        context_length=len(context),
        needle=needle,
    )


class NIAHBenchmark:
    """
    Needle in a Haystack Benchmark.
    
    Generates test cases with varying:
    - Context length (8K to 1M+ chars)
    - Needle position (start, middle, end)
    - Needle type (number, fact, code)
    """
    
    def __init__(
        self,
        context_lengths: List[int] = None,
        needle_positions: List[float] = None,
        needle_types: List[str] = None,
        samples_per_config: int = 1,
        seed: int = 42,
    ):
        """
        Initialize benchmark.
        
        Args:
            context_lengths: List of target context lengths
            needle_positions: List of needle positions (0.0 to 1.0)
            needle_types: List of needle types
            samples_per_config: Number of samples per configuration
            seed: Random seed
        """
        self.context_lengths = context_lengths or [10000, 50000, 100000, 500000]
        self.needle_positions = needle_positions or [0.1, 0.5, 0.9]
        self.needle_types = needle_types or ["number", "fact"]
        self.samples_per_config = samples_per_config
        self.seed = seed
    
    def generate_samples(self) -> List[NIAHSample]:
        """Generate all benchmark samples."""
        samples = []
        sample_idx = 0
        
        for length in self.context_lengths:
            for position in self.needle_positions:
                for needle_type in self.needle_types:
                    for i in range(self.samples_per_config):
                        sample = generate_niah_sample(
                            target_length=length,
                            needle_position=position,
                            needle_type=needle_type,
                            seed=self.seed + sample_idx,
                        )
                        samples.append(sample)
                        sample_idx += 1
        
        return samples
    
    def evaluate(self, predicted: str, expected: str) -> bool:
        """Check if prediction is correct."""
        # Normalize
        pred = predicted.lower().strip()
        exp = expected.lower().strip()
        
        # Exact match
        if pred == exp:
            return True
        
        # Check if expected is contained in predicted
        if exp in pred:
            return True
        
        return False
    
    def __len__(self):
        return (
            len(self.context_lengths) *
            len(self.needle_positions) *
            len(self.needle_types) *
            self.samples_per_config
        )


if __name__ == "__main__":
    # Test generation
    sample = generate_niah_sample(target_length=10000, needle_position=0.5)
    print(f"Context length: {sample.context_length}")
    print(f"Query: {sample.query}")
    print(f"Answer: {sample.answer}")
    print(f"Needle: {sample.needle}")
