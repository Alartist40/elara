"""
Elara CLI - Simplified Multi-Tier AI System.
"""

import argparse
import sys
import os
import logging
import asyncio
import numpy as np

def main():
    """
    Entry point that parses CLI options and runs Elara in text, interactive, or voice modes.
    
    Parses command-line flags (notably --text, --interactive, --voice, --voice-input, --voice-streaming,
    --tts-mimi, --tts-nemo, --tts-cpu), initializes tiered engines, router, safety filter, tool router,
    and voice gateway, then dispatches execution to one of three modes:
    - Voice input mode: starts an asynchronous full‑duplex voice conversation.
    - Interactive mode: enters a REPL that prints assistant responses (and optionally speaks them).
    - Text mode: processes a single text input, prints the response, and optionally speaks it.
    
    Side effects:
    - Loads environment variables from a .env file.
    - Constructs and mutates engine, router, safety, tools, and voice gateway instances.
    - May perform I/O to stdout and use audio output or the microphone depending on flags.
    """
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(description="Elara v2.0 - Functional AI")
    parser.add_argument("--text", type=str, help="Text input")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--voice", action="store_true", help="Use voice output")
    parser.add_argument("--voice-input", action="store_true", help="Use voice input (microphone)")
    parser.add_argument("--voice-streaming", action="store_true", help="Stream TTS for lower latency")
    parser.add_argument("--tts-mimi", action="store_true", default=True, help="Use Mimi neural TTS (default)")
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

    voice = VoiceGateway(
        tts_use_mimi=not (args.tts_nemo or args.tts_cpu),
        tts_use_nemo=args.tts_nemo,
    )
    safety = SafetyFilter()
    tools = ToolRouter()

    if args.voice_input:
        asyncio.run(voice_conversation_mode(args, tier1, tier2, tier3, router, safety, tools, voice))
    elif args.interactive:
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

async def voice_conversation_mode(args, tier1, tier2, tier3, router, safety, tools, voice):
    """
    Manage a full-duplex voice conversation that routes live audio through STT, persona-guided generation, safety and tool checks, and TTS playback.
    
    Sets up a voice persona, a duplex voice handler with user/assistant text and audio callbacks, and a recorder; starts the handler and streams captured audio chunks to it until interrupted (Ctrl+C). The handler forwards recognized text to the tiered generation pipeline via process_input using the persona's system prompt.
    
    Parameters:
        args: Parsed CLI arguments that can influence voice/tts configuration.
        tier1: Tier1Engine used for lightweight generation.
        tier2: Tier2Engine used for intermediate generation and fallback.
        tier3: Tier3Engine used for high-capability generation when available.
        router: TierRouter that selects which tier to use for a given input.
        safety: SafetyFilter used to pre- and post-check content.
        tools: ToolRouter used to execute external tools and provide contextual results.
        voice: VoiceGateway providing STT/TTS engines and configuration.
    """
    from elara_core.voice.duplex_handler import DuplexVoiceHandler
    from elara_core.persona.voice_persona import VoicePersonaManager
    from elara_core.voice.recorder import VoiceRecorder

    # Initialize persona
    persona = VoicePersonaManager(voice._mimi_tts)
    persona.load_persona("elara")

    # Create duplex handler
    handler = DuplexVoiceHandler(
        stt_engine=voice.stt or voice._ensure_stt() or voice.stt,
        process_callback=lambda text: process_input(
            text, tier1, tier2, tier3, router, safety, tools,
            system_prompt=persona.get_system_prompt()
        ),
        tts_engine=voice._mimi_tts or voice, # Fallback to voice gateway
        persona_manager=persona,
    )

    # Set up callbacks
    def on_user(text: str):
        """
        Prints the user's text prefixed with "User:" and re-displays the input prompt.
        
        Parameters:
            text (str): The user's utterance or transcribed speech to display.
        """
        print(f"\rUser: {text}")
        print("> ", end="", flush=True)

    def on_assistant(text: str):
        """
        Print the assistant's response to the console prefixed with "Elara:" and reprint the input prompt.
        
        Parameters:
            text (str): Assistant response text to display.
        """
        print(f"\rElara: {text}")
        print("> ", end="", flush=True)

    def on_audio(chunk: np.ndarray):
        # Play audio chunk (platform-specific)
        # Using sounddevice for playback as well
        """
        Play a raw audio chunk through the system audio output.
        
        Plays the provided NumPy array of PCM audio samples using sounddevice at a 24000 Hz sample rate.
        
        Parameters:
            chunk (np.ndarray): Audio samples (1-D or 2-D array) encoded as floating-point PCM suitable for playback at 24000 Hz.
        """
        import sounddevice as sd
        sd.play(chunk, samplerate=24000) # Mimi SR

    handler.on_user_text = on_user
    handler.on_assistant_text = on_assistant
    handler.on_audio_out = on_audio

    recorder = VoiceRecorder(sample_rate=16000) # Whisper SR

    print("Voice conversation started. Speak naturally (Ctrl+C to exit)...")
    await handler.start()

    try:
        async for chunk in recorder.stream():
            await handler.process_audio_chunk(chunk)
    except KeyboardInterrupt:
        await handler.stop()
        recorder.stop()
        print("\nConversation ended.")

def process_input(user_input, tier1, tier2, tier3, router, safety, tools, system_prompt=None):
    # 1. Safety Pre-check: Thread cleaned input through the pipeline
    """
    Process a user input through safety checks, optional tool execution, tier selection, and generation, then apply post-generation safety filtering.
    
    Parameters:
        system_prompt (str, optional): Optional system-level prompt passed to the generation engines to influence response tone or behavior.
    
    Returns:
        str: The assistant's final response text. May be:
          - the input scrubbed by the safety filter if the pre-check blocks the request,
          - the generated response from a tiered engine (possibly incorporating tool output context),
          - or the safety block message "Blocked by safety filter" if the post-check removes the generated content.
    """
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
        response = tier1.generate(prompt, system_prompt=system_prompt)
    elif tier == 2:
        response = tier2.generate(prompt, system_prompt=system_prompt)
    else:
        if tier3.is_available():
            response = tier3.generate(prompt, system_prompt=system_prompt)
        else:
            response = tier2.generate(prompt, system_prompt=system_prompt) # Fallback

    # 6. Safety Post-check
    allowed_post, final_response = safety.check(response)
    if not allowed_post:
        return final_response or "Blocked by safety filter"

    return final_response

if __name__ == "__main__":
    main()