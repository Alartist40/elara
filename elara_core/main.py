"""
Elara CLI - Simplified Multi-Tier AI System.
"""

import argparse
import sys
import os
import logging

def main():
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(description="Elara v2.0 - Functional AI")
    parser.add_argument("--text", type=str, help="Text input")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--voice", action="store_true", help="Use voice output")
    parser.add_argument("--tts-nemo", action="store_true", default=None, help="Force NeMo TTS (requires GPU)")
    parser.add_argument("--tts-cpu", action="store_true", help="Force CPU TTS (pyttsx3)")
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

    # Determine TTS preference
    use_nemo = None
    if args.tts_nemo:
        use_nemo = True
    elif args.tts_cpu:
        use_nemo = False

    voice = VoiceGateway(tts_use_nemo=use_nemo)
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
    try:
        tool_result = tools.execute(scrubbed_input)
    except Exception as e:
        logging.warning(f"Tool execution failed: {e}")
        tool_result = None

    # 3. Build context from tool output if available
    context = ""
    if tool_result and tool_result.success:
        # Truncate and clearly delimit tool output to mitigate prompt injection risks
        MAX_TOOL_OUTPUT = 500
        safe_output = str(tool_result.output)[:MAX_TOOL_OUTPUT]
        safe_name = str(tool_result.name)[:64].replace("\n", " ")
        context = (
            f"[TOOL RESULT — {safe_name}]\n"
            f"{safe_output}\n"
            f"[END TOOL RESULT]\n\n"
        )

    # 4. Tier selection
    tier = router.select_tier(scrubbed_input)

    # 5. Generation
    prompt = context + scrubbed_input if context else scrubbed_input

    if tier == 1:
        response = tier1.generate(prompt)
    elif tier == 2:
        response = tier2.generate(prompt)
    else:
        if tier3.is_available():
            response = tier3.generate(prompt)
        else:
            response = tier2.generate(prompt) # Fallback

    # 6. Safety Post-check
    allowed_post, final_response = safety.check(response)
    if not allowed_post:
        return final_response or "Blocked by safety filter"

    return final_response

if __name__ == "__main__":
    main()
