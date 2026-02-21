"""
Full-duplex voice conversation handler.
Allows interruptions, backchanneling, and natural turn-taking.
"""

import asyncio
import numpy as np
from collections import deque
from typing import Callable, Optional, Any
import logging

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
    ):
        """
        Initialize the full‑duplex conversation handler with STT, TTS, processing callback, and persona manager.
        
        Parameters:
            stt_engine: Speech-to-text engine used to transcribe user audio.
            process_callback: Callable that accepts transcribed user text and returns assistant response text.
            tts_engine: Text-to-speech engine used to synthesize assistant responses (may support streaming).
            persona_manager: Manager providing persona/context for responses.
            sample_rate (int): Audio sample rate in Hz for incoming PCM (default 16000).
        
        The constructor sets up audio framing (frame_size), runtime state flags (is_active, is_speaking),
        the current utterance buffer, a silence frame counter, and optional runtime callbacks:
        on_user_text, on_assistant_text, and on_audio_out.
        """
        self.stt = stt_engine
        self.process_callback = process_callback
        self.tts = tts_engine
        self.persona = persona_manager

        self.sample_rate = sample_rate
        self.frame_size = 1920  # 80ms at 24kHz, but we might need to adjust for STT

        # State
        self.is_active = False
        self.is_speaking = False
        self.current_utterance: list[np.ndarray] = []
        self._silence_frames = 0

        # Callbacks
        self.on_user_text: Optional[Callable[[str], None]] = None
        self.on_assistant_text: Optional[Callable[[str], None]] = None
        self.on_audio_out: Optional[Callable[[np.ndarray], None]] = None

    async def process_audio_chunk(self, pcm: np.ndarray):
        """
        Process a single microphone audio chunk, accumulate speech into the current utterance, and trigger interruption or utterance processing as needed.
        
        This method is a runtime entrypoint for incoming PCM audio frames: if the handler is not active the frame is ignored; otherwise a simple energy-based voice activity check determines whether the frame is treated as speech or silence. Speech frames are appended to the internal utterance buffer and silence frames increment an internal counter; if an incoming speech frame arrives while the assistant is speaking, an interruption handler is invoked; if prolonged silence follows buffered speech, the buffered utterance is dispatched for transcription and response generation.
        
        Parameters:
            pcm (np.ndarray): A single chunk of raw PCM audio samples (mono). The sample rate and frame size are determined by the handler instance.
        """
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
        """
        Detects whether an audio frame contains speech using mean absolute amplitude.
        
        Parameters:
            pcm (np.ndarray): Mono audio samples for a single frame.
            threshold (float): Mean absolute amplitude threshold; frames with mean absolute value greater than this are considered speech.
        
        Returns:
            `true` if the frame contains speech, `false` otherwise.
        """
        return np.abs(pcm).mean() > threshold

    async def _handle_interruption(self):
        """
        Handle an incoming user interruption during assistant speech.
        
        Marks the handler as not speaking and logs the event. Implementations may stop any in-progress TTS synthesis when called.
        """
        logger.info("Interruption detected")
        self.is_speaking = False
        # Stop current synthesis if possible

    async def _process_utterance(self):
        """
        Process the accumulated audio buffer: transcribe it to text, invoke the user-text callback if present, and trigger generation of the assistant response.
        
        The buffered audio is concatenated and cleared and the silence counter reset before transcription. If the transcription is empty or shorter than two characters, no response is generated.
        """
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
        """
        Generate and deliver the assistant's spoken response for the given user text.
        
        Sets the handler into a speaking state, obtains reply text via the configured process_callback, invokes on_assistant_text with the reply if present, and streams synthesized audio to on_audio_out. Uses tts.synthesize_streaming when available and falls back to tts.synthesize; synthesis stops early if the speaking state is cleared to allow user interruptions.
        
        Parameters:
            user_text (str): Transcribed user utterance to respond to.
        """
        self.is_speaking = True

        # Use the process callback to handle tools, safety, and routing
        response_text = self.process_callback(user_text)

        if self.on_assistant_text:
            self.on_assistant_text(response_text)

        # Stream TTS
        if hasattr(self.tts, 'synthesize_streaming'):
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
        """
        Activate the duplex voice handler so it begins accepting and processing incoming audio chunks.
        
        This sets internal state to allow process_audio_chunk to handle incoming microphone frames and trigger speech detection, transcription, and response generation.
        """
        self.is_active = True

    async def stop(self):
        """
        Deactivate the handler so it stops accepting and processing incoming audio.
        
        After calling this, the handler will ignore subsequent audio chunks until restarted.
        """
        self.is_active = False