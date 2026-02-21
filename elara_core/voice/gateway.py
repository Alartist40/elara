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
        """
        Initialize a VoiceGateway to manage speech-to-text and text-to-speech backends.
        
        Parameters:
            stt_model (str): Whisper STT model size identifier to use (e.g., "tiny").
            tts_use_nemo (Optional[bool]): If True, prefer NeMo-based TTS; if False, disable NeMo; if None, automatically detect availability.
            tts_use_mimi (bool): If True, prefer Mimi TTS when available; otherwise prefer the configured non-Mimi TTS.
            device (str): Device selection for models (e.g., "cpu", "cuda", or "auto").
        
        Description:
            Records configuration and prepares lazy-initialized placeholders for STT and TTS backends.
        """
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

    def _ensure_stt(self) -> None:
        """
        Ensure the speech-to-text backend is initialized.
        
        If an STT instance is not already present, create a WhisperSTT using the gateway's configured model size and device.
        """
        if self.stt is None:
            self.stt = WhisperSTT(self.stt_model_size, self.device)

    def _ensure_tts(self) -> None:
        """
        Ensure a TTS backend is initialized for this VoiceGateway instance.
        
        If a TTS backend is already set, the method does nothing. When Mimi usage is enabled, it attempts to initialize the streaming Mimi TTS backend and uses it if successful; otherwise it falls back to creating an ElaraTTS instance according to the configured NeMo preference and device. The chosen backend is stored on the instance as either `_mimi_tts` (for Mimi) or `tts` (for ElaraTTS).
        """
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
        """
        Transcribe spoken audio input to text.
        
        Parameters:
            audio_input (str | bytes | numpy.ndarray): Path to an audio file, raw audio bytes, or a PCM audio array.
        
        Returns:
            transcription (str): The transcribed text.
        """
        self._ensure_stt()
        return self.stt.transcribe(audio_input)

    def synthesize(self, text: str, **kwargs) -> np.ndarray:
        """
        Synthesize speech from the given text and return the resulting PCM audio as a NumPy array.
        
        Returns:
            np.ndarray: 1-D array of PCM samples; an empty array if synthesis produced no audio.
        """
        res = self.speak(text)
        return res if res is not None else np.zeros(0)

    def speak(self, text: str, output_path: Optional[Path] = None) -> Optional[np.ndarray]:
        """
        Synthesize spoken audio for the given text using the configured TTS backend.
        
        Parameters:
            text (str): The text to synthesize.
            output_path (Optional[Path]): If provided, write the synthesized audio to this file and return `None`. When omitted, return the synthesized PCM samples.
        
        Returns:
            Optional[numpy.ndarray]: PCM audio samples as a NumPy array when not writing to a file, or `None` if the audio was written to disk.
        """
        self._ensure_tts()

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
        """
        Stream TTS output as consecutive PCM audio chunks for real-time playback.
        
        Parameters:
            text (str): Text to synthesize.
        
        Returns:
            PCM audio chunks as 1-D numpy arrays of samples; each yielded value is the next chunk in playback order.
            
        Notes:
            If a streaming Mimi backend is available, chunks are produced by that backend. Otherwise the full PCM is synthesized first and then yielded in fixed-size frames.
        """
        self._ensure_tts()

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
        """
        Return runtime status and configuration for the voice gateway.
        
        Returns:
            dict: Mapping with keys:
                - `stt_loaded` (bool): `True` if the speech-to-text backend is initialized, `False` otherwise.
                - `tts_loaded` (bool): `True` if the text-to-speech backend is initialized, `False` otherwise.
                - `stt_model` (str): Configured STT model size identifier.
                - `device` (str): Configured device string (e.g., "auto", "cpu", "cuda").
                - `tts_use_nemo` (Optional[bool]): Flag indicating whether Nemo-based TTS is preferred (`True`/`False`) or `None` if undetermined.
        """
        return {
            "stt_loaded": self.stt is not None,
            "tts_loaded": self.tts is not None,
            "stt_model": self.stt_model_size,
            "device": self.device,
            "tts_use_nemo": self.tts_use_nemo,
        }