import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
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
        index_path: str = "data/faiss.index",
        docs_path: str = "data/documents.json",
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

        # FAISS index (pre-built or empty)
        self.index_path = Path(index_path)
        self.docs_path = Path(docs_path)

        if self.index_path.exists() and self.docs_path.exists():
            self.index = faiss.read_index(str(index_path))
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

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """Get top-k relevant documents."""
        if len(self.documents) == 0 or self.encoder is None:
            return []

        # Encode query
        query_emb = self.encoder.encode([query])
        faiss.normalize_L2(query_emb)

        # Search
        scores, indices = self.index.search(query_emb, k)

        # Return documents (filter -1 for empty results)
        return [
            self.documents[i]
            for i in indices[0]
            if i != -1 and i < len(self.documents)
        ]

    def generate(self, query: str, max_tokens: int = 512) -> str:
        if self.generator is None:
            docs = self.retrieve(query, k=3)
            if not docs:
                return "Error: No generator available and no documents retrieved."

            return f"""Retrieved documents (no generator to synthesize answer):

{chr(10).join(f"{i+1}. {d[:200]}..." for i, d in enumerate(docs))}

Query: {query}
[Connect Tier 1 engine to synthesize answers]"""

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

        return self.generator.generate(prompt, max_tokens)

    def has_relevant_docs(self, query: str, threshold: float = 0.3) -> bool:
        """Check if query has relevant documents (for routing)."""
        if len(self.documents) == 0 or self.encoder is None:
            return False

        query_emb = self.encoder.encode([query])
        faiss.normalize_L2(query_emb)
        scores, _ = self.index.search(query_emb, 1)

        return scores[0][0] > threshold
