import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import json
import logging
import functools
from pathlib import Path
from typing import Optional, Any

class Tier2Engine:
    """
    Retrieval-Augmented Generation.
    Not CLaRa. Not learned compression.
    Just FAISS + context injection.
    """

    def __init__(
        self,
        tier1_engine: Optional[Any] = None,  # Reuse for generation
        index_path: Optional[str] = None,
        docs_path: Optional[str] = None,
        model_name: str = "all-MiniLM-L6-v2",  # 22MB embedding model
    ):
        self.generator = tier1_engine

        # Embedding model (CPU, fast)
        try:
            self.encoder = SentenceTransformer(model_name)
            dim = self.encoder.get_sentence_embedding_dimension()
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            self.encoder = None
            dim = 384 # Fallback

        # Instance-specific cache to avoid memory leaks with @lru_cache on methods
        self._setup_cache()

        # FAISS index (pre-built or empty)
        self.index_path = Path(index_path or os.getenv("ELARA_INDEX_PATH", "data/faiss.index"))
        self.docs_path = Path(docs_path or os.getenv("ELARA_DOCS_PATH", "data/documents.json"))
        self.documents = []

        try:
            if self.index_path.exists() and self.docs_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                with open(self.docs_path, encoding="utf-8") as f:
                    self.documents = json.load(f)

                # Verify index consistency
                if self.index.ntotal != len(self.documents):
                    raise ValueError(f"Index size ({self.index.ntotal}) mismatch with documents ({len(self.documents)})")
            else:
                self.index = faiss.IndexFlatIP(dim)
        except Exception as e:
            logging.error(f"FAISS index corrupted or mismatch: {e}")
            self.index = faiss.IndexFlatIP(dim)
            if self.docs_path.exists():
                self._rebuild_index()
            else:
                self.documents = []

    def _rebuild_index(self):
        """
        Rebuilds the FAISS index and internal document list from the JSON file at self.docs_path.

        If the JSON file contains no documents, clears the internal documents list and leaves the index empty. On error while reading or rebuilding, clears the internal documents list and logs a critical error.
        """
        try:
            logging.info("Attempting to rebuild index from documents.json...")
            with open(self.docs_path, encoding="utf-8") as f:
                texts = json.load(f)

            if not texts:
                self.documents = []
                return

            self.documents = [] # Clear and re-add
            self.add_documents(texts)
            logging.info("Index rebuilt successfully")
        except Exception as e:
            logging.critical(f"Index rebuild failed: {e}")
            self.documents = []

    def add_documents(self, texts: list[str]):
        """
        Add the given texts to the FAISS-backed document store and persist the updated index and document list to disk.

        Each text is encoded into an embedding, normalized for cosine similarity, appended to the in-memory FAISS index and documents list, and then both the index file and documents JSON are written to their configured paths.

        Parameters:
            texts (list[str]): Documents to add to the store.

        Note:
            Call this during setup, not at runtime.

        Raises:
            RuntimeError: If the encoder is not loaded.
        """
        if self.encoder is None:
            raise RuntimeError("Encoder not loaded.")

        # Encode to vectors
        embeddings = self.encoder.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Add to index
        self.index.add(embeddings)
        self.documents.extend(texts)

        # Save
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.docs_path, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False)

    def _setup_cache(self):
        """
        Create and attach an instance-scoped cached query-to-embedding function.

        Sets self._get_cached_embedding to an LRU-cached callable that accepts a query string and returns a normalized float32 NumPy embedding suitable for cosine-similarity searches, or None if the encoder is not available. The cache holds up to 128 entries.
        """
        encoder = self.encoder
        @functools.lru_cache(maxsize=128)
        def get_emb(query: str):
            """
            Encode a text query and return its L2-normalized embedding using the configured encoder.

            Parameters:
                query (str): Text to be embedded.

            Returns:
                numpy.ndarray | None: The normalized float32 embedding for query, or None if no encoder is available.
            """
            if encoder is None:
                return None
            # Encode and normalize
            emb = encoder.encode([query], convert_to_numpy=True).astype(np.float32)
            faiss.normalize_L2(emb)
            return emb
        self._get_cached_embedding = get_emb

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        """
        Retrieve the most relevant documents for a query using the FAISS index.

        Returns:
            list[str]: Up to k documents ranked by relevance. Returns an empty list if there are no stored documents, the encoder is unavailable, or the query cannot be embedded.
        """
        if len(self.documents) == 0 or self.encoder is None:
            return []

        # Get cached embedding
        query_emb = self._get_cached_embedding(query)
        if query_emb is None:
            return []

        # Search
        scores, indices = self.index.search(query_emb, k)

        # Return documents (filter -1 for empty results)
        return [
            self.documents[i]
            for i in indices[0]
            if i != -1 and i < len(self.documents)
        ]

    def generate_standalone(self, query: str) -> str:
        """Extractive retrieval fallback used when tier1 is unavailable. max_tokens does not apply."""
        docs = self.retrieve(query, k=3)
        if not docs:
            return "I don't have relevant information to answer that. [Tier 1 unavailable]"

        # Simple extractive answer - pick most relevant sentence
        best_doc = docs[0]
        sentences = best_doc.split('.')
        excerpt = sentences[0].strip() + '.' if sentences else best_doc[:200]

        return (
            f"[Retrieved from documents - Tier 1 unavailable]\n\n"
            f"Relevant excerpt: {excerpt}\n\n"
            f"For a complete answer, please ensure the local language model is loaded."
        )

    def generate(self, query: str, max_tokens: int = 512, system_prompt: Optional[str] = None) -> str:
        if self.generator is None:
            logging.warning("Tier2Engine: No generator available. Using standalone fallback.")
            return self.generate_standalone(query)

        # Retrieve context
        docs = self.retrieve(query, k=3)

        # Build augmented prompt
        if docs:
            context = "\n\n".join([f"Document {i+1}: {d}" for i, d in enumerate(docs)])
            prompt = f"""Use the following documents to answer the question.

{context}

Question: {query}
Answer:"""
        else:
            # No relevant docs, fall back to Tier 1 style
            prompt = query

        return self.generator.generate(prompt, max_tokens, system_prompt=system_prompt)

    def has_relevant_docs(self, query: str, threshold: float = 0.3) -> bool:
        """Check if query has relevant documents (for routing)."""
        if len(self.documents) == 0 or self.encoder is None:
            return False

        query_emb = self._get_cached_embedding(query)
        if query_emb is None:
            return False

        scores, _ = self.index.search(query_emb, 1)

        return scores[0][0] > threshold
