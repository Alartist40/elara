"""
Elara CLI - Simplified Multi-Tier AI System.
"""

import argparse
import sys
import os
from dotenv import load_dotenv

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Elara v2.0 - Functional AI")
    parser.add_argument("--text", type=str, help="Text input")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--voice", action="store_true", help="Use voice output")
    args = parser.parse_args()

    if not (args.interactive or args.text):
        parser.print_help()
        return

    # Defer heavy component initialization
    from elara_core.tiers.tier1 import Tier1Engine
    from elara_core.tiers.tier2 import Tier2Engine
    from elara_core.tiers.tier3 import Tier3Engine
    from elara_core.tiers.router import TierRouter
    from elara_core.voice.gateway import VoiceGateway
    from elara_core.safety.filter import SafetyFilter
    from elara_core.tools.router import ToolRouter

    tier1 = Tier1Engine()
    tier2 = Tier2Engine(tier1)
    tier3 = Tier3Engine()
    router = TierRouter(tier2)
    voice = VoiceGateway()
    safety = SafetyFilter()
    tools = ToolRouter()

    if args.interactive:
        print("Elara v2.0 Functional - Type 'exit' to quit")
        while True:
            try:
                user_input = input("> ").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if user_input.lower() in ["exit", "quit"]:
                break

            response = process_input(user_input, tier1, tier2, tier3, router, safety, tools)
            print(f"Assistant: {response}")

            if args.voice:
                voice.speak(response)
    elif args.text:
        response = process_input(args.text, tier1, tier2, tier3, router, safety, tools)
        print(response)
        if args.voice:
            voice.speak(response)

def process_input(user_input, tier1, tier2, tier3, router, safety, tools):
    # 1. Safety Pre-check: Thread cleaned input through the pipeline
    allowed, scrubbed_input = safety.check(user_input)
    if not allowed:
        return scrubbed_input

    # 2. Tool routing
    tool_result = tools.execute(scrubbed_input)
    if tool_result and tool_result.success:
        return f"Calculated: {tool_result.output}"

    # 3. Tier selection
    tier = router.select_tier(scrubbed_input)

    # 4. Generation
    if tier == 1:
        response = tier1.generate(scrubbed_input)
    elif tier == 2:
        response = tier2.generate(scrubbed_input)
    else:
        if tier3.is_available():
            response = tier3.generate(scrubbed_input)
        else:
            response = tier2.generate(scrubbed_input) # Fallback

    # 5. Safety Post-check
    allowed_post, final_response = safety.check(response)
    if not allowed_post:
        return final_response or "Blocked by safety filter"

    return final_response

if __name__ == "__main__":
    main()
