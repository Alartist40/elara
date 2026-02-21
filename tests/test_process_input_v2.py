import unittest
from unittest.mock import MagicMock
from elara_core.main import process_input

class TestProcessInput(unittest.TestCase):
    def setUp(self):
        self.tier1 = MagicMock()
        self.tier2 = MagicMock()
        self.tier3 = MagicMock()
        self.router = MagicMock()
        self.safety = MagicMock()
        self.tools = MagicMock()

    def test_process_input_tool_success(self):
        # Mock safety to allow input
        self.safety.check.side_effect = lambda x: (True, x)

        # Mock tool to succeed
        tool_result = MagicMock()
        tool_result.success = True
        tool_result.name = "calculator"
        tool_result.output = "4"
        self.tools.execute.return_value = tool_result

        # Mock router to select Tier 1
        self.router.select_tier.return_value = 1

        # Mock Tier 1 to return a response
        self.tier1.generate.return_value = "The result is 4."

        response = process_input("calculate 2+2", self.tier1, self.tier2, self.tier3, self.router, self.safety, self.tools)

        # Verify tool result was used as context with NEW FORMAT
        expected_context = (
            "[TOOL RESULT — calculator]\n"
            "4\n"
            "[END TOOL RESULT]\n\n"
        )
        self.tier1.generate.assert_called_with(expected_context + "calculate 2+2", system_prompt=None)
        self.assertEqual(response, "The result is 4.")

    def test_process_input_tool_exception(self):
        # Mock safety to allow input
        self.safety.check.side_effect = lambda x: (True, x)

        # Mock tool to raise exception
        self.tools.execute.side_effect = Exception("Crash!")

        # Mock router to select Tier 1
        self.router.select_tier.return_value = 1
        self.tier1.generate.return_value = "Normal response"

        response = process_input("some query", self.tier1, self.tier2, self.tier3, self.router, self.safety, self.tools)

        self.assertEqual(response, "Normal response")
        self.tier1.generate.assert_called_with("some query", system_prompt=None)

if __name__ == "__main__":
    unittest.main()
