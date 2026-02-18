"""
Elara CLI - Main entry point for the Elara multi-modal AI system.
"""

import argparse
import sys
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Elara v2.0 - Multi-Modal AI System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m elara_core.main --text "Hello, how are you?"
  python -m elara_core.main --text "Analyze this" --tier 3
  python -m elara_core.main --status
  python -m elara_core.main --load-tiers 1 2 3
        """,
    )

    parser.add_argument("--config", type=str, default="config/system_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--text", type=str, help="Text input for generation")
    parser.add_argument("--audio", type=str, help="Path to audio file for voice input")
    parser.add_argument("--voice-output", action="store_true",
                        help="Output as audio")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3],
                        help="Force a specific tier")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--status", action="store_true",
                        help="Show system status")
    parser.add_argument("--load-tiers", nargs="+", type=int,
                        help="Load specific tier models")
    parser.add_argument("--interactive", action="store_true",
                        help="Enter interactive chat mode")

    args = parser.parse_args()

    from elara_core.tiered.engine import TieredInferenceEngine

    engine = TieredInferenceEngine(config_path=args.config)

    # Show status
    if args.status:
        status = engine.get_system_status()
        print(json.dumps(status, indent=2, default=str))
        return

    # Load models
    if args.load_tiers:
        print(f"Loading tier models: {args.load_tiers}")
        result = engine.load_tier_models(args.load_tiers)
        for component, success in result.items():
            symbol = "✓" if success else "✗"
            print(f"  {symbol} {component}")
        return

    # Interactive mode
    if args.interactive:
        _interactive_mode(engine, args)
        return

    # Single generation
    if args.text:
        result = engine.generate(
            input_data=args.text,
            voice_output=args.voice_output,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            force_tier=args.tier,
        )
        print(f"\n{'='*60}")
        print(f"Tier: {result.metadata.get('tier', '?')}")
        print(f"Latency: {result.metadata.get('latency_ms', 0):.1f}ms")
        print(f"{'='*60}")
        print(result.text or "[Audio output generated]")
        return

    if args.audio:
        result = engine.generate(
            input_data=args.audio,
            voice_input=True,
            voice_output=args.voice_output,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            force_tier=args.tier,
        )
        print(result.text or "[Audio output generated]")
        return

    parser.print_help()


def _interactive_mode(engine, args):
    """Interactive chat loop."""
    print("=" * 60)
    print("  Elara v2.0 - Interactive Mode")
    print("  Type 'quit' to exit, 'status' for system info")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if user_input.lower() == "status":
            status = engine.get_system_status()
            print(json.dumps(status, indent=2, default=str))
            continue

        result = engine.generate(
            input_data=user_input,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            force_tier=args.tier,
        )

        tier = result.metadata.get("tier", "?")
        latency = result.metadata.get("latency_ms", 0)
        blocked = result.metadata.get("blocked", False)

        if blocked:
            print(f"\n⛔ [Blocked] {result.text}")
        else:
            print(f"\n🤖 Elara (Tier {tier}, {latency:.0f}ms): {result.text}")


if __name__ == "__main__":
    main()
