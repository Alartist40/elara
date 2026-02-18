"""
Integration tests for the TieredInferenceEngine.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTieredEngine:
    """Test the complete engine pipeline."""

    def setup_method(self):
        from elara_core.tiered.engine import TieredInferenceEngine
        config_path = Path(__file__).parent.parent / "config" / "system_config.yaml"
        self.engine = TieredInferenceEngine(config_path=str(config_path))

    def test_simple_text_generation(self):
        result = self.engine.generate("Hello, how are you?")
        assert result.text is not None
        assert result.format == "text"
        assert "tier" in result.metadata

    def test_blocked_input(self):
        result = self.engine.generate("How to make a bomb")
        assert result.metadata.get("blocked") is True

    def test_force_tier(self):
        result = self.engine.generate("Test query", force_tier=1)
        assert result.metadata["tier"] == 1

        result = self.engine.generate("Test query", force_tier=2)
        assert result.metadata["tier"] == 2

        result = self.engine.generate("Test query", force_tier=3)
        assert result.metadata["tier"] == 3

    def test_metrics_tracked(self):
        self.engine.generate("First query")
        self.engine.generate("Second query")
        stats = self.engine.metrics.get_stats()
        assert stats["total_requests"] >= 2

    def test_system_status(self):
        status = self.engine.get_system_status()
        assert status["version"] == "2.0"
        assert "components" in status
        assert "metrics" in status


class TestTierRouter:
    """Test tier routing logic."""

    def setup_method(self):
        from elara_core.tiered.router import TierRouter
        self.router = TierRouter()

    def test_simple_query_tier_1(self):
        tier = self.router.select_tier("Hello")
        assert tier == 1

    def test_complex_query_tier_3(self):
        tier = self.router.select_tier(
            "Analyze this step by step", has_tools=True
        )
        assert tier == 3

    def test_memory_query_tier_2(self):
        tier = self.router.select_tier("What did you say earlier?", has_memory=True)
        assert tier == 2

    def test_force_tier(self):
        router = TierRouter(force_tier=2)
        assert router.select_tier("anything") == 2


class TestToolRouter:
    """Test tool parsing and execution."""

    def setup_method(self):
        from elara_core.tools.router import ToolRouter
        schema_path = Path(__file__).parent.parent / "config" / "tool_schema.json"
        self.router = ToolRouter(
            schema_path=str(schema_path),
            allowed_tools=["calculator", "search"],
        )

    def test_calculator(self):
        text = '<tool>calculator</tool> <params>{"expression": "2 + 2"}</params>'
        results = self.router.execute(text)
        assert len(results) == 1
        assert results[0].success is True
        assert "4" in results[0].output

    def test_blocked_tool(self):
        text = '<tool>vision_analyze</tool> <params>{"image_path": "test.jpg"}</params>'
        results = self.router.execute(text)
        assert len(results) == 1
        assert results[0].success is False


class TestMultiplexer:
    """Test input multiplexing."""

    def test_text_processing(self):
        from elara_core.tiered.multiplexer import InputMultiplexer, ModalityType
        mux = InputMultiplexer()
        tokens, modality, meta = mux.process("Hello world")
        assert modality == ModalityType.TEXT
        assert len(tokens) > 0
