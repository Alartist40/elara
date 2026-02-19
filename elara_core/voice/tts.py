"""
TTS Engine with NeMo and pyttsx3 fallback.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import warnings

class TTSProvider:
    """Base class for TTS providers."""
    def synthesize(self, text: str, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def synthesize_to_file(self, text: str, output_path: Path, **kwargs) -> None:
        raise NotImplementedError

class NeMoTTS(TTSProvider):
    """NVIDIA NeMo TTS: FastPitch + HiFi-GAN."""
    def __init__(self, fastpitch_path=None, hifigan_path=None, speaker_id=0, device="auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.speaker_id = speaker_id
        self.sample_rate = 22050
        self.fastpitch = None
        self.vocoder = None
        self._fastpitch_path = fastpitch_path
        self._hifigan_path = hifigan_path
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded: return
        from nemo.collections.tts.models import FastPitchModel, HifiGanModel
        if self._fastpitch_path:
            self.fastpitch = FastPitchModel.restore_from(str(self._fastpitch_path)).to(self.device)
        else:
            self.fastpitch = FastPitchModel.from_pretrained(model_name="tts_en_fastpitch").to(self.device)

        if self._hifigan_path:
            self.vocoder = HifiGanModel.restore_from(str(self._hifigan_path)).to(self.device)
        else:
            self.vocoder = HifiGanModel.from_pretrained(model_name="tts_hifigan").to(self.device)

        self.fastpitch.eval()
        self.vocoder.eval()
        self._loaded = True

    def synthesize(self, text: str, pace: float = 1.0, **kwargs) -> np.ndarray:
        self._ensure_loaded()
        with torch.no_grad():
            parsed = self.fastpitch.parse(text)
            spectrogram = self.fastpitch.generate_spectrogram(tokens=parsed, speaker=self.speaker_id, pace=pace)
            audio = self.vocoder.convert_spectrogram_to_audio(spec=spectrogram)
        return audio.squeeze().cpu().numpy()

    def synthesize_to_file(self, text: str, output_path: Path, **kwargs) -> None:
        import soundfile as sf
        audio = self.synthesize(text, **kwargs)
        sf.write(str(output_path), audio, self.sample_rate)

class PyTTSx3(TTSProvider):
    """Offline TTS fallback using pyttsx3."""
    def __init__(self, rate=150):
        import pyttsx3
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)

    def synthesize(self, text: str, **kwargs) -> np.ndarray:
        # pyttsx3 doesn't easily return a numpy array directly without saving to file first
        # For simplicity, we'll return an empty array if not saving to file,
        # or implement a temporary file read if really needed.
        # But usually in Elara, it's used with synthesize_to_file for playback.
        warnings.warn("PyTTSx3.synthesize returns empty array. Use synthesize_to_file.")
        return np.zeros(0)

    def synthesize_to_file(self, text: str, output_path: Path, **kwargs) -> None:
        self.engine.save_to_file(text, str(output_path))
        self.engine.runAndWait()

class ElaraTTS:
    """Wrapper that tries NeMo and falls back to pyttsx3."""
    def __init__(self, use_nemo=True, **kwargs):
        self.provider = None
        if use_nemo:
            try:
                self.provider = NeMoTTS(**kwargs)
                # Try to load (optional check)
            except Exception as e:
                print(f"Failed to load NeMo TTS, falling back to pyttsx3: {e}")
                self.provider = PyTTSx3()
        else:
            self.provider = PyTTSx3()

    def synthesize(self, text: str, **kwargs) -> np.ndarray:
        return self.provider.synthesize(text, **kwargs)

    def synthesize_to_file(self, text: str, output_path: Path, **kwargs) -> None:
        self.provider.synthesize_to_file(text, output_path, **kwargs)
