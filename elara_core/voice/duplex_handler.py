"""
Full-duplex voice conversation handler.
Allows interruptions, backchanneling, and natural turn-taking.
"""

import asyncio
import numpy as np
from collections import deque
from typing import Callable, Optional
import logging
from elara_core.voice.piper_tts import PiperTTS

logger = logging.getLogger(__name__)

class DuplexVoiceHandler:
    """
    Manages full-duplex audio I/O for natural conversation.
    """

    def __init__(
        self,
        stt_engine,
        process_callback,
        tts_engine,
        persona_manager,
        sample_rate: int = 16000, # STT usually expects 16kHz
        use_streaming: bool = True,
    ):
        """
        Initialize the duplex voice conversation handler with speech recognition, response processing, text-to-speech, and persona management; configure runtime settings, internal state, and optional callbacks.

        Parameters:
            stt_engine: Speech-to-text engine instance used for transcribing captured audio.
            process_callback: Callable that accepts a user-transcribed string and returns the assistant's response text.
            tts_engine: Text-to-speech engine used to synthesize assistant audio output.
            persona_manager: Manager or configuration object that controls assistant persona or context.
            sample_rate (int): Audio sample rate for incoming PCM frames (default 16000). STT engines commonly expect 16 kHz.
            use_streaming (bool): If True, prefer streaming TTS output when supported by the TTS engine; otherwise use full-buffer synthesis.

        Behavior:
            Stores provided engines and callbacks, sets frame sizing from MimiTTS.FRAME_SIZE, initializes activity/speaking flags, the current utterance buffer, silence counters, and placeholder attributes for optional runtime callbacks (on_user_text, on_assistant_text, on_audio_out, on_interrupt).
        """
        self.stt = stt_engine
        self.process_callback = process_callback
        self.tts = tts_engine
        self.persona = persona_manager
        self.use_streaming = use_streaming

        self.sample_rate = sample_rate
        self.frame_size = PiperTTS.FRAME_SIZE  # 1024

        # State
        self.is_active = False
        self.is_speaking = False
        self.current_utterance: list[np.ndarray] = []
        self._silence_frames = 0

        # Callbacks
        self.on_user_text: Optional[Callable[[str], None]] = None
        self.on_assistant_text: Optional[Callable[[str], None]] = None
        self.on_audio_out: Optional[Callable[[np.ndarray], None]] = None
        self.on_interrupt: Optional[Callable[[], None]] = None

    async def process_audio_chunk(self, pcm: np.ndarray):
        """Feed incoming audio from microphone."""
        if not self.is_active:
            return

        # Simple VAD (energy-based)
        if self._is_speech(pcm):
            self.current_utterance.append(pcm)
            self._silence_frames = 0

            # Check for interruption
            if self.is_speaking:
                await self._handle_interruption()
        else:
            if self.current_utterance:
                self._silence_frames += 1
                self.current_utterance.append(pcm)

                # Silence - check if we have a complete utterance (~800ms of silence)
                if self._silence_frames > 10:
                    await self._process_utterance()

    def _is_speech(self, pcm: np.ndarray, threshold: float = 0.02) -> bool:
        """Simple energy-based VAD."""
        return np.abs(pcm).mean() > threshold

    async def _handle_interruption(self):
        """User interrupted assistant."""
        logger.info("Interruption detected")
        self.is_speaking = False
        if self.on_interrupt:
            self.on_interrupt()

    async def _process_utterance(self):
        """Transcribe and respond to user speech."""
        utterance = np.concatenate(self.current_utterance)
        self.current_utterance = []
        self._silence_frames = 0

        # STT (Whisper expects 16kHz)
        text = self.stt.transcribe(utterance)
        if not text or len(text.strip()) < 2:
            return

        if self.on_user_text:
            self.on_user_text(text)

        # Get response
        await self._generate_response(text)

    async def _generate_response(self, user_text: str):
        """Generate and stream assistant response."""
        self.is_speaking = True

        # Use the process callback to handle tools, safety, and routing
        response_text = self.process_callback(user_text)

        if self.on_assistant_text:
            self.on_assistant_text(response_text)

        # Stream TTS
        if self.use_streaming and hasattr(self.tts, 'synthesize_streaming'):
            async for chunk in self.tts.synthesize_streaming(response_text):
                if not self.is_speaking:
                    break  # Interrupted

                if self.on_audio_out:
                    self.on_audio_out(chunk)
        else:
            # Fallback: generate then play
            pcm = self.tts.synthesize(response_text)
            if self.is_speaking and self.on_audio_out:
                self.on_audio_out(pcm)

        self.is_speaking = False

    async def start(self):
        self.is_active = True

    async def stop(self):
        self.is_active = False
