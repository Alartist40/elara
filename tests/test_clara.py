"""
Tests for CLaRa components.
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSCPCompressor:
    """Test SCP document compression."""

    def setup_method(self):
        from elara_core.clara.compressor import SCPCompressor, SCPCompressorConfig
        self.config = SCPCompressorConfig(
            vocab_size=1000, d_model=64, n_layers=2,
            n_heads=4, d_head=16, d_ff=128,
            n_memory_tokens=4, compression_rate=4,
        )
        self.compressor = SCPCompressor(self.config)

    def test_forward_shape(self):
        doc_ids = torch.randint(0, 1000, (2, 32))
        memory = self.compressor(doc_ids)
        assert memory.shape == (2, 4, 64)  # [B, n_memory_tokens, d_model]

    def test_scp_loss(self):
        doc_ids = torch.randint(0, 1000, (2, 32))
        target_ids = torch.randint(0, 1000, (2, 16))
        losses = self.compressor.compute_scp_loss(doc_ids, target_ids)
        assert "loss" in losses
        assert losses["loss"].requires_grad


class TestDifferentiableTopK:
    """Test differentiable top-k selection."""

    def test_selection(self):
        from elara_core.clara.topk import DifferentiableTopK
        topk = DifferentiableTopK(k=3, temperature=0.01)
        sims = torch.randn(2, 10)
        indices, weights = topk(sims)
        assert indices.shape == (2, 3)
        assert weights.shape == (2, 3, 10)


class TestCLaRaStore:
    """Test document store operations."""

    def setup_method(self):
        from elara_core.clara.store import CLaRaStore
        self.store = CLaRaStore(store_path="/tmp/test_clara_store", d_model=64, n_memory_tokens=4)
        self.store.clear()

    def test_add_and_retrieve(self):
        docs = torch.randn(5, 4, 64)
        self.store.add_documents(docs)
        assert len(self.store) == 5

        query = torch.randn(1, 4, 64)
        retrieved, scores, meta = self.store.retrieve(query, top_k=3)
        assert retrieved.shape[1] == 3
        assert scores.shape == (1, 3)

    def test_empty_store_raises(self):
        with pytest.raises(ValueError):
            self.store.retrieve(torch.randn(1, 4, 64))

    def test_stats(self):
        docs = torch.randn(3, 4, 64)
        self.store.add_documents(docs)
        stats = self.store.get_stats()
        assert stats["doc_count"] == 3


class TestQueryReasoner:
    """Test query encoding."""

    def test_forward_shape(self):
        from elara_core.clara.query_reasoner import QueryReasoner
        from elara_core.clara.compressor import SCPCompressorConfig
        cfg = SCPCompressorConfig(vocab_size=1000, d_model=64, n_layers=2,
                                   n_heads=4, d_head=16, d_ff=128, n_memory_tokens=4)
        qr = QueryReasoner(cfg)
        query_ids = torch.randint(0, 1000, (2, 10))
        embedding = qr(query_ids)
        assert embedding.shape == (2, 4, 64)
