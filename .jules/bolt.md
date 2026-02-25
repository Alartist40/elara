## 2025-01-30 - Memoized Query Embeddings in RAG
**Learning:** In a multi-tier routing architecture, the same query is often processed by multiple components (e.g., the router and then the selected tier). Encoding the query into an embedding is a significant CPU bottleneck (~10ms for MiniLM). Memoizing these embeddings at the engine level avoids redundant transformer passes.
**Action:** Always check if a heavy computation like embedding generation is repeated within the same request lifecycle and implement caching.

## 2025-01-30 - Avoiding lru_cache Memory Leaks on Instance Methods
**Learning:** Using `@functools.lru_cache` directly on instance methods is an anti-pattern because the cache (at the class level) holds strong references to `self`, preventing garbage collection.
**Action:** Use per-instance caches by initializing them in `__init__` or using a closure that captures only the necessary non-self state.

## 2025-01-31 - Redundant computations in generator expressions
**Learning:** Calling a string manipulation method like `.lower()` inside a generator expression passed to `any()` or `all()` can result in that method being called multiple times (once for each iteration until short-circuit).
**Action:** Always hoist common computations (like lowercasing the search target) outside of the generator expression to ensure they are performed only once.

## 2026-02-25 - PyTorch Broadcasting over Explicit Repeating
**Learning:** Using `.repeat()` to expand a conditioning tensor (e.g., a voice embedding) to match the sequence length of another tensor (e.g., prosody) is inefficient. PyTorch's broadcasting mechanism handles this automatically during arithmetic operations without allocating a large intermediate tensor.
**Action:** Prefer broadcasting for expanding tensors along singleton dimensions to save memory and CPU cycles.
