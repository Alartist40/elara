import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import json
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
        index_path: Optional[str] = None,
        docs_path: Optional[str] = None,
        model_name: str = "all-MiniLM-L6-v2",  # 22MB embedding model
    ):
        """
        Initialize the Tier2Engine, preparing embedding encoder, FAISS index, and document store.
        
        Parameters:
        	tier1_engine (Optional[Any]): Optional generator to delegate text generation to.
        	index_path (Optional[str]): Path to the FAISS index file; if None, resolved from ELARA_INDEX_PATH or defaults to "data/faiss.index".
        	docs_path (Optional[str]): Path to the documents JSON file; if None, resolved from ELARA_DOCS_PATH or defaults to "data/documents.json".
        	model_name (str): SentenceTransformer model name to use for embeddings.
        
        Notes:
        	- Attempts to load the specified SentenceTransformer model; on failure, `self.encoder` is set to `None` and a fallback embedding dimension is used.
        	- If both index and document files exist at the resolved paths, the FAISS index and documents are loaded; otherwise an empty FAISS IndexFlatIP and an empty documents list are created.
        """
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
        self.index_path = Path(index_path or os.getenv("ELARA_INDEX_PATH", "data/faiss.index"))
        self.docs_path = Path(docs_path or os.getenv("ELARA_DOCS_PATH", "data/documents.json"))

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
        """
        Retrieve the top relevant documents for a query.
        
        If no documents are stored or the encoder is not initialized, returns an empty list.
        
        Returns:
        	A list of up to `k` documents ranked by relevance for the given query; an empty list if no relevant documents are found.
        """
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

    def generate_standalone(self, query: str, max_tokens: int = 512) -> str:
        """
        Produce a concise answer using the most relevant retrieved document when no external generator is configured.
        
        If up to three documents are retrieved, return the first sentence of the top-ranked document (including a trailing period). If no documents are found, return a default message indicating lack of information.
        
        Parameters:
        	query (str): The user query used to retrieve relevant documents.
        	max_tokens (int): Maximum token budget for the generated answer; currently unused and reserved for compatibility.
        
        Returns:
        	A short answer string: the first sentence of the most relevant document if available, otherwise a default "no information" message.
        """
        docs = self.retrieve(query, k=3)
        if not docs:
            return "I don't have relevant information to answer that."

        # Simple extractive answer - pick most relevant sentence
        best_doc = docs[0]
        sentences = best_doc.split('.')
        return sentences[0] + '.' if sentences else best_doc

    def generate(self, query: str, max_tokens: int = 512, system_prompt: Optional[str] = None) -> str:
        """
        Generate an answer to a query using retrieved documents and an underlying generator.
        
        If relevant documents are found, they are included as context in the prompt sent to the generator; otherwise the raw query is used. If no generator is configured, a standalone fallback is used that returns a concise answer from the most relevant document.
        
        Parameters:
            query (str): The user question to answer.
            max_tokens (int): Maximum number of tokens for the generated answer.
            system_prompt (Optional[str]): Optional system-level instruction passed through to the underlying generator.
        
        Returns:
            str: The generated answer.
        """
        if self.generator is None:
            logging.warning("Tier2Engine: No generator available. Using standalone fallback.")
            return self.generate_standalone(query, max_tokens)

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
        Determine whether there exists a document relevant to the given query above the similarity threshold.
        
        Parameters:
            query (str): Query text to match against stored documents.
            threshold (float): Minimum cosine-similarity score required to consider a document relevant (default 0.3).
        
        Returns:
            bool: `True` if the top retrieved document's similarity score is greater than `threshold`, `False` otherwise. Returns `False` when no documents are stored or the encoder is unavailable.
        """
        if len(self.documents) == 0 or self.encoder is None:
            return False

        query_emb = self.encoder.encode([query])
        faiss.normalize_L2(query_emb)
        scores, _ = self.index.search(query_emb, 1)

        return scores[0][0] > threshold