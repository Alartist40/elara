import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import json
import logging
import functools
from pathlib import Path
from typing import List, Optional, Any

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

        if self.index_path.exists() and self.docs_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            try:
                with open(self.docs_path, encoding="utf-8") as f:
                    self.documents = json.load(f)
            except Exception as e:
                print(f"Error loading documents: {e}")
                self.documents = []
        else:
            # Empty index - will build on first add
            self.index = faiss.IndexFlatIP(dim)
            self.documents = []

    def add_documents(self, texts: List[str]):
        """
        Add documents to the store.
        Call this during setup, not at runtime.
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
        """Initialize instance-specific query embedding cache."""
        encoder = self.encoder
        @functools.lru_cache(maxsize=128)
        def get_emb(query: str):
            if encoder is None:
                return None
            # Encode and normalize
            emb = encoder.encode([query], convert_to_numpy=True).astype(np.float32)
            faiss.normalize_L2(emb)
            return emb

        def get_emb_safe(query: str):
            res = get_emb(query)
            # Return a copy to prevent callers from corrupting the cache
            return res.copy() if res is not None else None

        self._get_cached_embedding = get_emb_safe

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """
        Get top-k relevant documents for a query.

        Parameters:
            query (str): The search query.
            k (int): Number of documents to retrieve.

        Returns:
            List[str]: List of relevant document texts.
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
            return "I don't have relevant information to answer that."

        # Simple extractive answer - pick most relevant sentence
        best_doc = docs[0]
        sentences = best_doc.split('.')
        return sentences[0] + '.' if sentences else best_doc

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
        """
        Check if the query has relevant documents in the index.

        This returns True if the highest similarity score exceeds the threshold,
        and False otherwise.

        Parameters:
            query (str): The search query.
            threshold (float): Similarity threshold (0.0 to 1.0).

        Returns:
            bool: True if relevant documents are found, False otherwise.
        """
        if len(self.documents) == 0 or self.encoder is None:
            return False

        query_emb = self._get_cached_embedding(query)
        if query_emb is None:
            return False

        scores, _ = self.index.search(query_emb, 1)

        return scores[0][0] > threshold
