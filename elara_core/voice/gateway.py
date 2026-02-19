"""
VoiceGateway - Unified interface for voice I/O.
Optimized for lazy loading.
"""

import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict, Any

from elara_core.voice.stt import WhisperSTT
from elara_core.voice.tts import ElaraTTS


class VoiceGateway:
    """
    Unified interface for voice I/O.
    """

    def __init__(
        self,
        stt_model: str = "tiny", # Default to tiny for speed
        tts_use_nemo: Optional[bool] = None, # Auto-detect if None
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
        self.stt_model_size = stt_model
        self.tts_use_nemo = tts_use_nemo
        self.device = device

    def _ensure_stt(self) -> None:
        if self.stt is None:
            self.stt = WhisperSTT(self.stt_model_size, self.device)

    def _ensure_tts(self) -> None:
        if self.tts is None:
            self.tts = ElaraTTS(use_nemo=self.tts_use_nemo, device=self.device)

    def listen(self, audio_input: Union[str, bytes, np.ndarray]) -> str:
        self._ensure_stt()
        return self.stt.transcribe(audio_input)

    def speak(self, text: str, output_path: Optional[Path] = None) -> Optional[np.ndarray]:
        self._ensure_tts()
        if output_path:
            self.tts.synthesize_to_file(text, output_path)
            return None
        else:
            return self.tts.synthesize(text)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "stt_loaded": self.stt is not None,
            "tts_loaded": self.tts is not None,
            "stt_model": self.stt_model_size,
            "device": self.device,
        }
