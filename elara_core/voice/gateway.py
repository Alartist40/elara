"""
VoiceGateway - Unified interface for voice I/O.
Optimized for lazy loading.
"""

import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict, Any

from elara_core.voice.stt import WhisperSTT
from elara_core.voice.tts import ElaraTTS
from elara_core.voice.mimi_tts import MimiTTS, StreamingMimiTTS


class VoiceGateway:
    """
    Unified interface for voice I/O.
    """

    def __init__(
        self,
        stt_model: str = "tiny", # Default to tiny for speed
        tts_use_nemo: Optional[bool] = None, # Auto-detect if None
        tts_use_mimi: bool = True, # NEW: Prefer Mimi
        device: str = "auto",
    ):
        if tts_use_nemo is None:
            # Auto-detect: use NeMo if CUDA available
            try:
                import torch
                tts_use_nemo = torch.cuda.is_available()
            except ImportError:
                tts_use_nemo = False

        self.stt: Optional[WhisperSTT] = None
        self.tts: Optional[ElaraTTS] = None
        self._mimi_tts: Optional[MimiTTS] = None
        self.stt_model_size = stt_model
        self.tts_use_nemo = tts_use_nemo
        self.tts_use_mimi = tts_use_mimi
        self.device = device

    def ensure_stt(self) -> WhisperSTT:
        if self.stt is None:
            self.stt = WhisperSTT(self.stt_model_size, self.device)
        return self.stt

    def get_mimi(self) -> Optional[MimiTTS]:
        self.ensure_tts()
        return self._mimi_tts

    def ensure_tts(self) -> None:
        if self.tts is not None or self._mimi_tts is not None:
            return

        # Try Mimi first
        if self.tts_use_mimi:
            try:
                self._mimi_tts = StreamingMimiTTS(device=None if self.device == "auto" else self.device)
                return
            except Exception as e:
                print(f"Mimi TTS failed to load: {e}, falling back...")

        # Fallback to existing NeMo/pyttsx3
        if self.tts is None:
            self.tts = ElaraTTS(use_nemo=self.tts_use_nemo, device=self.device)

    def listen(self, audio_input: Union[str, bytes, np.ndarray]) -> str:
        stt = self.ensure_stt()
        return stt.transcribe(audio_input)

    def synthesize(self, text: str, **kwargs) -> np.ndarray:
        """Alias for speak() to maintain compatibility with TTS providers."""
        res = self.speak(text)
        return res if res is not None else np.zeros(0)

    def speak(self, text: str, output_path: Optional[Path] = None) -> Optional[np.ndarray]:
        self.ensure_tts()

        if self._mimi_tts:
            pcm = self._mimi_tts.synthesize(text)
            if output_path:
                import sphn
                sphn.write_wav(str(output_path), pcm, self._mimi_tts.SAMPLE_RATE)
                return None
            return pcm

        if output_path:
            self.tts.synthesize_to_file(text, output_path)
            return None
        else:
            return self.tts.synthesize(text)

    async def speak_streaming(self, text: str):
        """Streaming TTS for real-time playback."""
        self.ensure_tts()

        if isinstance(self._mimi_tts, StreamingMimiTTS):
            async for chunk in self._mimi_tts.synthesize_streaming(text):
                yield chunk
        else:
            # Fallback: generate all then yield in chunks
            pcm = self.speak(text)
            frame_size = 1920
            for i in range(0, len(pcm), frame_size):
                yield pcm[i:i+frame_size]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "stt_loaded": self.stt is not None,
            "tts_loaded": self.tts is not None,
            "stt_model": self.stt_model_size,
            "device": self.device,
            "tts_use_nemo": self.tts_use_nemo,
        }
