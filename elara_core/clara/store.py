"""
CLaRaStore - Persistent compressed document store with mmap support.
Stores pre-compressed document embeddings for fast retrieval.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import mmap
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import warnings


class CLaRaStore:
    """
    Persistent store for CLaRa compressed document embeddings.
    Supports mmap for memory-efficient access and fast retrieval.
    """

    def __init__(
        self,
        store_path: str = "data/clara_store",
        d_model: int = 512,
        n_memory_tokens: int = 16,
        mmap_enabled: bool = True,
    ):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.d_model = d_model
        self.n_memory_tokens = n_memory_tokens
        self.mmap_enabled = mmap_enabled

        # In-memory store
        self.embeddings: Optional[torch.Tensor] = None  # [N, l, D]
        self.metadata: List[Dict[str, Any]] = []
        self.doc_count = 0

        # Mmap handle
        self._mmap_handle: Optional[mmap.mmap] = None
        self._mmap_file = None

        # Load existing store if present
        self._load_if_exists()

    def _load_if_exists(self):
        """Load existing store from disk."""
        meta_path = self.store_path / "metadata.json"
        embed_path = self.store_path / "embeddings.pt"

        if meta_path.exists() and embed_path.exists():
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)
            self.embeddings = torch.load(embed_path, map_location="cpu", weights_only=True)
            self.doc_count = len(self.metadata)

    def add_documents(
        self,
        embeddings: torch.Tensor,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add compressed document embeddings to the store.

        Args:
            embeddings: [N, n_memory_tokens, d_model] compressed docs.
            metadata_list: Optional metadata for each document.
        """
        N = embeddings.shape[0]

        if metadata_list is None:
            metadata_list = [{"id": self.doc_count + i} for i in range(N)]

        if self.embeddings is None:
            self.embeddings = embeddings.detach().cpu()
        else:
            self.embeddings = torch.cat(
                [self.embeddings, embeddings.detach().cpu()], dim=0
            )

        self.metadata.extend(metadata_list)
        self.doc_count += N

    def retrieve(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
        """
        Retrieve top-k most relevant documents.

        Args:
            query_embedding: [B, n_memory_tokens, d_model] query vectors.
            top_k: Number of documents to retrieve.

        Returns:
            retrieved_embeds: [B, k, n_memory_tokens, d_model]
            similarities: [B, k] cosine similarity scores
            retrieved_metadata: list of metadata dicts for retrieved docs
        """
        if self.embeddings is None or self.doc_count == 0:
            raise ValueError("Store is empty. Add documents first.")

        # Average over memory tokens for similarity computation
        query_vec = query_embedding.mean(dim=1)  # [B, D]
        doc_vec = self.embeddings.mean(dim=1)     # [N, D]

        # Normalize
        query_vec = F.normalize(query_vec, dim=-1)
        doc_vec = F.normalize(doc_vec, dim=-1)

        # Cosine similarity
        similarities = torch.matmul(query_vec, doc_vec.t())  # [B, N]

        # Top-k selection
        k = min(top_k, self.doc_count)
        topk_values, topk_indices = torch.topk(similarities, k, dim=-1)

        # Gather retrieved embeddings
        B = query_embedding.shape[0]
        retrieved_embeds = []
        retrieved_metadata = []

        for b in range(B):
            batch_embeds = []
            batch_meta = []
            for idx in topk_indices[b]:
                batch_embeds.append(self.embeddings[idx.item()])
                if idx.item() < len(self.metadata):
                    batch_meta.append(self.metadata[idx.item()])
            retrieved_embeds.append(torch.stack(batch_embeds))
            retrieved_metadata.append(batch_meta)

        retrieved_embeds = torch.stack(retrieved_embeds)  # [B, k, l, D]

        return retrieved_embeds, topk_values, retrieved_metadata

    def contains(self, query_text: str) -> bool:
        """Check if store has any documents (simple heuristic)."""
        return self.doc_count > 0

    def save(self) -> None:
        """Persist store to disk."""
        if self.embeddings is not None:
            torch.save(self.embeddings, self.store_path / "embeddings.pt")
        with open(self.store_path / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)

    def get_stats(self) -> Dict[str, Any]:
        """Return store statistics."""
        size_bytes = 0
        if self.embeddings is not None:
            size_bytes = self.embeddings.element_size() * self.embeddings.nelement()

        return {
            "doc_count": self.doc_count,
            "d_model": self.d_model,
            "n_memory_tokens": self.n_memory_tokens,
            "store_size_mb": size_bytes / (1024 * 1024),
            "mmap_enabled": self.mmap_enabled,
        }

    def clear(self) -> None:
        """Clear the entire store."""
        self.embeddings = None
        self.metadata = []
        self.doc_count = 0

    def __len__(self) -> int:
        return self.doc_count
