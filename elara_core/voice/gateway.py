"""
VoiceGateway - Unified interface for voice I/O.
Optimized for lazy loading.
"""

import numpy as np
from pathlib import Path
from typing import Union, Optional, Any

from elara_core.voice.olmo_stt import OLMoSTT
from elara_core.voice.tts import ElaraTTS
from elara_core.voice.piper_tts import PiperTTS, StreamingPiperTTS


class VoiceGateway:
    """
    Unified interface for voice I/O.
    """

    def __init__(
        self,
        stt_model: str = "tiny",
        tts_use_nemo: Optional[bool] = None,
        tts_use_mimi: bool = True,  # Keep flag name for compatibility
        device: str = "auto",
    ):
        if tts_use_nemo is None:
            try:
                import torch
                tts_use_nemo = torch.cuda.is_available()
            except ImportError:
                tts_use_nemo = False

        self.stt: Optional[OLMoSTT] = None
        self.tts: Optional[ElaraTTS] = None
        self._piper_tts: Optional[PiperTTS] = None
        self.stt_model_size = stt_model
        self.tts_use_nemo = tts_use_nemo
        self.tts_use_mimi = tts_use_mimi  # Controls whether to try Piper first
        self.device = device

    def ensure_stt(self) -> OLMoSTT:
        if self.stt is None:
            # Map model names to OLMoASR sizes
            size_map = {
                "tiny": "tiny",
                "base": "base",
                "small": "small",
                "medium": "medium",
                "large": "large",
                "large-v2": "large-v2"
            }
            olmo_size = size_map.get(self.stt_model_size, "base")
            self.stt = OLMoSTT(olmo_size, self.device)
        return self.stt

    def get_piper(self) -> Optional[PiperTTS]:
        self.ensure_tts()
        return self._piper_tts

    def ensure_tts(self) -> None:
        if self.tts is not None or self._piper_tts is not None:
            return

        # Try Piper first
        if self.tts_use_mimi:
            try:
                self._piper_tts = StreamingPiperTTS()
                return
            except Exception as e:
                print(f"Piper TTS failed to load: {e}, falling back...")

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

        if self._piper_tts:
            pcm = self._piper_tts.synthesize(text)
            if output_path:
                import soundfile as sf
                sf.write(str(output_path), pcm, self._piper_tts.SAMPLE_RATE)
                return None
            # Play audio through speakers
            try:
                import sounddevice as sd
                sd.play(pcm, samplerate=self._piper_tts.SAMPLE_RATE)
                sd.wait()
            except Exception as e:
                print(f"Audio playback failed: {e}")
            return pcm

        if output_path:
            self.tts.synthesize_to_file(text, output_path)
            return None
        else:
            pcm = self.tts.synthesize(text)
            if pcm is not None and len(pcm) > 0:
                try:
                    import sounddevice as sd
                    sd.play(pcm, samplerate=22050)
                    sd.wait()
                except Exception as e:
                    print(f"Audio playback failed: {e}")
            return pcm

    async def speak_streaming(self, text: str):
        """
        Provide an async stream of PCM audio frames for synthesizing the given text.
        """
        self.ensure_tts()

        if isinstance(self._piper_tts, StreamingPiperTTS):
            async for chunk in self._piper_tts.synthesize_streaming(text):
                yield chunk
        else:
            # Fallback: generate all then yield in chunks
            pcm = self.speak(text)
            frame_size = PiperTTS.FRAME_SIZE
            for i in range(0, len(pcm), frame_size):
                yield pcm[i:i + frame_size]

    def get_stats(self) -> dict[str, Any]:
        """
        Provide runtime status and configuration of the VoiceGateway instance.
        """
        return {
            "stt_loaded": self.stt is not None,
            "tts_loaded": self.tts is not None or self._piper_tts is not None,
            "stt_model": self.stt_model_size,
            "stt_backend": "OLMoASR",
            "device": self.device,
            "tts_use_nemo": self.tts_use_nemo,
            "tts_use_piper": self._piper_tts is not None,
        }
