import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import functools
import logging
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
        Add multiple texts to the vector store and persist them to disk.
        
        Parameters:
            texts (List[str]): A list of document strings to encode and store.
        
        Raises:
            RuntimeError: If the embedding encoder is not loaded.
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

    @functools.lru_cache(maxsize=16)
    def _get_cached_embedding(self, query: str) -> Optional[np.ndarray]:
        """
        Return a normalized embedding for the given query; results are memoized to avoid redundant encodings.
        
        Parameters:
            query (str): Text to encode.
        
        Returns:
            np.ndarray or None: L2-normalized embedding vector for the query, or `None` if the encoder is unavailable.
        """
        if self.encoder is None:
            return None
        # Encode with explicit numpy conversion
        emb = self.encoder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(emb)
        return emb

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """
        Retrieve the most relevant documents for a query.
        
        Uses a cached embedding and the FAISS index to find up to `k` documents ordered by relevance. Returns an empty list when no documents are indexed or when an embedding cannot be produced.
        
        Parameters:
        	query (str): The input query to search for.
        	k (int): Maximum number of documents to return.
        
        Returns:
        	List[str]: The top relevant documents ordered by decreasing relevance; may contain fewer than `k` items.
        """
        if len(self.documents) == 0:
            return []

        # Use cached embedding to avoid redundant encoding overhead
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

    def generate(self, query: str, max_tokens: int = 512) -> str:
        if self.generator is None:
            logging.warning("Tier2Engine: returning raw documents — no generator available.")
            docs = self.retrieve(query, k=3)
            if not docs:
                return "Error: No generator available and no documents retrieved."

            doc_lines = chr(10).join(
                f"{i+1}. {d[:200]}{'...' if len(d) > 200 else ''}"
                for i, d in enumerate(docs)
            )
            return f"Retrieved documents (no generator available):\n\n{doc_lines}\n\nQuery: {query}"

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
        """
        Determine whether the FAISS index contains documents relevant to a query.
        
        This returns `false` when the index is empty or an embedding cannot be produced for the query.
        
        Parameters:
        	query (str): The user's query to check for relevant documents.
        	threshold (float): Similarity threshold in [0, 1]; the function considers a document relevant if the top similarity score is greater than this value.
        
        Returns:
        	`true` if the highest similarity score for the query exceeds `threshold`, `false` otherwise.
        """
        if len(self.documents) == 0:
            return False

        # Use cached embedding to avoid redundant encoding overhead
        query_emb = self._get_cached_embedding(query)
        if query_emb is None:
            return False

        scores, _ = self.index.search(query_emb, 1)

        return scores[0][0] > threshold