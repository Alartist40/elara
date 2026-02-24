## 2025-01-30 - Memoized Query Embeddings in RAG
**Learning:** In a multi-tier RAG system, the same user query is often passed through the embedding model multiple times (e.g., once for routing/relevance check and once for final retrieval). For CPU-bound edge devices, this BERT encoding is the primary bottleneck (~10-20ms per pass), while the vector search itself is negligible (<0.1ms). Memoizing these embeddings at the engine level avoids redundant transformer passes.
**Action:** Always memoize the query embedding within the RAG engine for the duration of a request lifecycle. Using an instance-specific cache provides a ~50% reduction in total RAG overhead.

## 2025-01-30 - Avoiding lru_cache Memory Leaks on Instance Methods
**Learning:** Using `@functools.lru_cache` directly on instance methods is an anti-pattern because the cache (at the class level) holds strong references to `self`, preventing garbage collection.
**Action:** Use per-instance caches by initializing them in `__init__` or using a closure that captures only the necessary non-self state.

## 2025-01-31 - Redundant computations in generator expressions
**Learning:** Calling a string manipulation method like `.lower()` inside a generator expression passed to `any()` or `all()` can result in that method being called multiple times (once for each iteration until short-circuit).
**Action:** Always hoist common computations (like lowercasing the search target) outside of the generator expression to ensure they are performed only once.
