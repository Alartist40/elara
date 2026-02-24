## 2025-05-14 - [RAG Query Encoding Caching]
**Learning:** In a multi-tier RAG system, the same user query is often passed through the embedding model multiple times (e.g., once for routing/relevance check and once for final retrieval). For CPU-bound edge devices, this BERT encoding is the primary bottleneck (~10-20ms per pass), while the vector search itself is negligible (<0.1ms).
**Action:** Always memoize the query embedding within the RAG engine for the duration of a request lifecycle.

**Refinement:** Avoid `functools.lru_cache` on instance methods to prevent memory leaks (holding references to `self`) and shared cache budget issues across instances. Use a per-instance `dict` instead. Always return a `.copy()` of cached mutable objects like `np.ndarray` to prevent accidental cache corruption by callers.
