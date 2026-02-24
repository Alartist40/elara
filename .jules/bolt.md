## 2025-01-30 - Memoized Query Embeddings in RAG
**Learning:** In a multi-tier routing architecture, the same query is often processed by multiple components (e.g., the router and then the selected tier). Encoding the query into an embedding is a significant CPU bottleneck (~10ms for MiniLM). Memoizing these embeddings at the engine level avoids redundant transformer passes.
**Action:** Always check if a heavy computation like embedding generation is repeated within the same request lifecycle and implement caching.

## 2025-01-30 - Avoiding lru_cache Memory Leaks on Instance Methods
**Learning:** Using `@functools.lru_cache` directly on instance methods is an anti-pattern because the cache (at the class level) holds strong references to `self`, preventing garbage collection.
**Action:** Use per-instance caches by initializing them in `__init__` or using a closure that captures only the necessary non-self state.
