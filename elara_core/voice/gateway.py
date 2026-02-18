"""
VoiceGateway - Unified interface for voice I/O.
Completely separate from core AI model.
"""

import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict, Any

from elara_core.voice.stt import WhisperSTT
from elara_core.voice.tts import NeMoTTS


class VoiceGateway:
    """
    Unified interface for voice I/O.
    Handles both Speech-to-Text (Whisper) and Text-to-Speech (NeMo).
    Completely separate from core AI model pipeline.
    """

    def __init__(
        self,
        stt_model: str = "base",
        tts_speaker: int = 0,
        device: str = "auto",
    ):
        self.stt: Optional[WhisperSTT] = None
        self.tts: Optional[NeMoTTS] = None
        self.stt_model_size = stt_model
        self.tts_speaker = tts_speaker
        self.device = device
        self._stt_initialized = False
        self._tts_initialized = False

    def ensure_stt(self) -> None:
        """Lazy load STT model."""
        if not self._stt_initialized:
            self.stt = WhisperSTT(self.stt_model_size, self.device)
            self._stt_initialized = True

    def ensure_tts(self) -> None:
        """Lazy load TTS model."""
        if not self._tts_initialized:
            self.tts = NeMoTTS(speaker_id=self.tts_speaker, device=self.device)
            self._tts_initialized = True

    def listen(
        self,
        audio_input: Union[str, bytes, np.ndarray],
        language: str = "en",
    ) -> str:
        """
        Audio → Text (STT).

        Args:
            audio_input: File path, bytes, or numpy array.
            language: ISO language code.

        Returns:
            Transcribed text string.
        """
        self.ensure_stt()
        return self.stt.transcribe(audio_input, language=language)

    def speak(
        self,
        text: str,
        pace: float = 1.0,
        output_path: Optional[Path] = None,
    ) -> Optional[np.ndarray]:
        """
        Text → Audio (TTS).

        Args:
            text: Text to speak.
            pace: Speaking rate multiplier.
            output_path: If provided, save to file instead of returning array.

        Returns:
            Audio array if output_path is None, else None.
        """
        self.ensure_tts()

        if output_path:
            self.tts.synthesize_to_file(text, output_path, pace=pace)
            return None
        else:
            return self.tts.synthesize(text, pace=pace)

    def get_stats(self) -> Dict[str, Any]:
        """Return voice component statistics."""
        return {
            "stt_loaded": self._stt_initialized,
            "tts_loaded": self._tts_initialized,
            "stt_model": self.stt_model_size if self._stt_initialized else None,
            "tts_speaker": self.tts_speaker if self._tts_initialized else None,
            "device": self.device,
        }
