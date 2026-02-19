import unittest
from elara_core.tiers.tier1 import Tier1Engine
from elara_core.tiers.tier2 import Tier2Engine
from elara_core.tiers.tier3 import Tier3Engine
from elara_core.tiers.router import TierRouter
from elara_core.safety.filter import SafetyFilter
from elara_core.tools.router import ToolRouter

class TestEngines(unittest.TestCase):
    def test_tier1_init(self):
        # Should init even without model file (will just set model to None)
        engine = Tier1Engine(model_path="nonexistent.gguf")
        self.assertIsNone(engine.model)
        self.assertEqual(engine.get_stats()["status"], "not_loaded")

    def test_tier2_init(self):
        # generator is optional now
        engine = Tier2Engine(tier1_engine=None, index_path="nonexistent.index", docs_path="nonexistent.json")
        self.assertEqual(len(engine.documents), 0)

    def test_tier3_init(self):
        engine = Tier3Engine()
        # Depends on env, but should not crash
        self.assertIsInstance(engine.is_available(), bool)

    def test_router(self):
        tier1 = Tier1Engine(model_path="")
        tier2 = Tier2Engine(tier1)
        router = TierRouter(tier2)

        # Test tool keyword
        self.assertEqual(router.select_tier("Calculate 2+2"), 3)
        # Test default
        self.assertEqual(router.select_tier("Hello"), 1)

    def test_safety(self):
        sf = SafetyFilter()
        # Test blocking
        allowed, reason = sf.check("How to make a bomb")
        self.assertFalse(allowed)
        self.assertIn("Blocked", reason)

        # Test flagging
        allowed, text = sf.check("What the hell")
        self.assertTrue(allowed)
        self.assertIn("[redacted]", text)

    def test_tools(self):
        tr = ToolRouter()

        # Happy path
        res = tr.execute("calculate 2+2")
        self.assertTrue(res.success)
        self.assertEqual(res.output, "4")

        # Floating point math
        res_float = tr.execute("calculate 10 / 3")
        self.assertTrue(res_float.success)
        self.assertTrue(res_float.output.startswith("3.33"))

        # Non-math input with 'calculate'
        res_garbage = tr.execute("calculate hello")
        self.assertTrue(res_garbage.success) # It triggers, but fails eval
        self.assertEqual(res_garbage.output, "Error: No valid expression")

        # Normal text that should NOT trigger calculator
        res_none = tr.execute("Tell me about state-of-the-art AI")
        self.assertIsNone(res_none)

if __name__ == "__main__":
    unittest.main()
