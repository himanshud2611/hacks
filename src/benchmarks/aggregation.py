"""
Aggregation Benchmark

Tests ability to aggregate information across many chunks.
Inspired by the OOLONG benchmark from the RLM paper.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class AggregationSample:
    """A single aggregation test sample."""
    context: List[str]  # List of text chunks
    query: str
    answer: str
    query_type: str
    context_length: int
    num_chunks: int
    relevant_chunks: List[int]  # Indices of chunks containing relevant info


def generate_counting_sample(
    num_chunks: int = 50,
    chunk_size: int = 2000,
    target_items: int = 5,
    seed: Optional[int] = None,
) -> AggregationSample:
    """
    Generate a counting aggregation sample.
    
    Task: Count occurrences of specific items across chunks.
    """
    if seed:
        random.seed(seed)
    
    # Categories and items
    categories = {
        "animals": ["dog", "cat", "elephant", "tiger", "penguin", "dolphin", "eagle", "bear"],
        "colors": ["red", "blue", "green", "yellow", "purple", "orange", "pink", "cyan"],
        "fruits": ["apple", "banana", "orange", "grape", "mango", "kiwi", "peach", "plum"],
    }
    
    category = random.choice(list(categories.keys()))
    items = categories[category]
    target_item = random.choice(items)
    
    # Generate chunks with random mentions
    chunks = []
    relevant_indices = []
    total_count = 0
    
    for i in range(num_chunks):
        # Generate filler text
        words = [random.choice(["the", "a", "an", "some", "many", "few", "several", "various"]) + " " +
                 random.choice(["report", "document", "analysis", "study", "review", "paper"]) + " " +
                 random.choice(["shows", "indicates", "reveals", "demonstrates", "suggests"])
                 for _ in range(chunk_size // 50)]
        
        chunk_text = ' '.join(words)
        
        # Maybe add target item mentions
        if random.random() < target_items / num_chunks * 2:  # Increase probability
            mentions = random.randint(1, 3)
            total_count += mentions
            relevant_indices.append(i)
            
            # Insert mentions
            for _ in range(mentions):
                mention = f" [Observation: Found a {target_item} in this section.] "
                pos = random.randint(0, len(chunk_text) - 1)
                chunk_text = chunk_text[:pos] + mention + chunk_text[pos:]
        
        chunks.append(f"=== Section {i+1} ===\n{chunk_text}\n")
    
    # Ensure we have at least some relevant chunks
    if total_count == 0:
        # Force add some mentions
        for _ in range(target_items):
            idx = random.randint(0, num_chunks - 1)
            relevant_indices.append(idx)
            mention = f" [Observation: Found a {target_item} in this section.] "
            chunks[idx] = chunks[idx] + mention
            total_count += 1
    
    query = f"How many times is '{target_item}' mentioned across all sections? Count all occurrences."
    answer = str(total_count)
    
    return AggregationSample(
        context=chunks,
        query=query,
        answer=answer,
        query_type="counting",
        context_length=sum(len(c) for c in chunks),
        num_chunks=num_chunks,
        relevant_chunks=list(set(relevant_indices)),
    )


def generate_comparison_sample(
    num_chunks: int = 20,
    chunk_size: int = 1000,
    seed: Optional[int] = None,
) -> AggregationSample:
    """
    Generate a comparison aggregation sample.
    
    Task: Find the maximum/minimum value across chunks.
    """
    if seed:
        random.seed(seed)
    
    entities = ["Company A", "Company B", "Company C", "Company D", "Company E",
                "Region X", "Region Y", "Region Z", "Division Alpha", "Division Beta"]
    metrics = ["revenue", "profit", "growth rate", "customer count", "market share"]
    
    entity = random.choice(entities)
    metric = random.choice(metrics)
    
    chunks = []
    values = []
    relevant_indices = []
    
    for i in range(num_chunks):
        # Generate report chunk
        chunk_text = f"=== Report Section {i+1} ===\n"
        chunk_text += f"This section covers various business metrics and analysis.\n"
        
        # Add some random data
        for _ in range(3):
            rand_entity = random.choice(entities)
            rand_metric = random.choice(metrics)
            rand_value = random.randint(10, 1000)
            chunk_text += f"- {rand_entity} {rand_metric}: {rand_value}\n"
        
        # Maybe add target entity data
        if random.random() < 0.4 or len(relevant_indices) == 0 and i == num_chunks - 1:
            value = random.randint(100, 10000)
            values.append(value)
            relevant_indices.append(i)
            chunk_text += f"\n** {entity} {metric}: {value} **\n"
        
        chunk_text += "\n" + "Lorem ipsum " * (chunk_size // 20) + "\n"
        chunks.append(chunk_text)
    
    if not values:
        # Ensure at least one value
        value = random.randint(100, 10000)
        values.append(value)
        relevant_indices.append(0)
        chunks[0] += f"\n** {entity} {metric}: {value} **\n"
    
    max_value = max(values)
    query = f"What is the highest {metric} reported for {entity} across all sections?"
    answer = str(max_value)
    
    return AggregationSample(
        context=chunks,
        query=query,
        answer=answer,
        query_type="comparison",
        context_length=sum(len(c) for c in chunks),
        num_chunks=num_chunks,
        relevant_chunks=relevant_indices,
    )


def generate_aggregation_sample(
    query_type: str = "counting",
    num_chunks: int = 50,
    chunk_size: int = 2000,
    seed: Optional[int] = None,
) -> AggregationSample:
    """Generate an aggregation sample of specified type."""
    if query_type == "counting":
        return generate_counting_sample(num_chunks, chunk_size, seed=seed)
    elif query_type == "comparison":
        return generate_comparison_sample(num_chunks, chunk_size, seed=seed)
    else:
        raise ValueError(f"Unknown query_type: {query_type}")


class AggregationBenchmark:
    """
    Aggregation Benchmark.
    
    Tests ability to:
    - Count occurrences across chunks
    - Find max/min values
    - Aggregate information semantically
    """
    
    def __init__(
        self,
        num_chunks_list: List[int] = None,
        chunk_sizes: List[int] = None,
        query_types: List[str] = None,
        samples_per_config: int = 2,
        seed: int = 42,
    ):
        self.num_chunks_list = num_chunks_list or [20, 50, 100]
        self.chunk_sizes = chunk_sizes or [1000, 2000]
        self.query_types = query_types or ["counting", "comparison"]
        self.samples_per_config = samples_per_config
        self.seed = seed
    
    def generate_samples(self) -> List[AggregationSample]:
        """Generate all benchmark samples."""
        samples = []
        sample_idx = 0
        
        for num_chunks in self.num_chunks_list:
            for chunk_size in self.chunk_sizes:
                for query_type in self.query_types:
                    for i in range(self.samples_per_config):
                        sample = generate_aggregation_sample(
                            query_type=query_type,
                            num_chunks=num_chunks,
                            chunk_size=chunk_size,
                            seed=self.seed + sample_idx,
                        )
                        samples.append(sample)
                        sample_idx += 1
        
        return samples
    
    def evaluate(self, predicted: str, expected: str) -> bool:
        """Check if prediction is correct."""
        # Extract numbers from prediction
        import re
        pred_numbers = re.findall(r'\d+', predicted)
        exp_numbers = re.findall(r'\d+', expected)
        
        if not pred_numbers or not exp_numbers:
            return predicted.strip() == expected.strip()
        
        # Check if any predicted number matches expected
        return any(p == exp_numbers[0] for p in pred_numbers)
    
    def __len__(self):
        return (
            len(self.num_chunks_list) *
            len(self.chunk_sizes) *
            len(self.query_types) *
            self.samples_per_config
        )


if __name__ == "__main__":
    # Test generation
    sample = generate_counting_sample(num_chunks=10, chunk_size=500, seed=42)
    print(f"Query: {sample.query}")
    print(f"Answer: {sample.answer}")
    print(f"Num chunks: {sample.num_chunks}")
    print(f"Context length: {sample.context_length}")
    print(f"Relevant chunks: {sample.relevant_chunks}")
