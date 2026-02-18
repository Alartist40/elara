"""
NeMoTTS - NVIDIA NeMo TTS wrapper (FastPitch + HiFi-GAN).
High-quality, low-latency, local inference.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import warnings


class NeMoTTS:
    """
    NVIDIA NeMo TTS: FastPitch + HiFi-GAN.
    High-quality, low-latency, local inference.
    """

    def __init__(
        self,
        fastpitch_path: Optional[Path] = None,
        hifigan_path: Optional[Path] = None,
        speaker_id: int = 0,
        device: str = "auto",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.speaker_id = speaker_id
        self.sample_rate = 22050  # HiFi-GAN default

        self.fastpitch = None
        self.vocoder = None
        self._fastpitch_path = fastpitch_path
        self._hifigan_path = hifigan_path
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy load NeMo models."""
        if self._loaded:
            return

        try:
            from nemo.collections.tts.models import FastPitchModel, HifiGanModel
        except ImportError:
            raise ImportError(
                "NeMo not installed. Run: pip install nemo_toolkit['all']"
            )

        # Load or download FastPitch
        if self._fastpitch_path:
            self.fastpitch = FastPitchModel.restore_from(
                str(self._fastpitch_path)
            ).to(self.device)
        else:
            self.fastpitch = FastPitchModel.from_pretrained(
                model_name="tts_en_fastpitch"
            ).to(self.device)

        # Load or download HiFi-GAN vocoder
        if self._hifigan_path:
            self.vocoder = HifiGanModel.restore_from(
                str(self._hifigan_path)
            ).to(self.device)
        else:
            self.vocoder = HifiGanModel.from_pretrained(
                model_name="tts_hifigan"
            ).to(self.device)

        self.fastpitch.eval()
        self.vocoder.eval()
        self._loaded = True

    def synthesize(
        self,
        text: str,
        pace: float = 1.0,
        pitch_shift: float = 0.0,
    ) -> np.ndarray:
        """
        Synthesize text to audio.

        Args:
            text: Input text (punctuation affects pauses).
            pace: Speaking rate multiplier (0.5=slow, 2.0=fast).
            pitch_shift: Pitch adjustment in semitones.

        Returns:
            Audio as numpy array (float32, 22.05kHz).
        """
        self._ensure_loaded()

        with torch.no_grad():
            parsed = self.fastpitch.parse(text)
            spectrogram = self.fastpitch.generate_spectrogram(
                tokens=parsed,
                speaker=self.speaker_id,
                pace=pace,
            )

            # Vocoder: spectrogram → audio
            audio = self.vocoder.convert_spectrogram_to_audio(spec=spectrogram)

        return audio.squeeze().cpu().numpy()

    def synthesize_to_file(
        self,
        text: str,
        output_path: Path,
        **kwargs,
    ) -> None:
        """Synthesize and save to WAV file."""
        try:
            import soundfile as sf
        except ImportError:
            raise ImportError("soundfile required: pip install soundfile")

        audio = self.synthesize(text, **kwargs)
        sf.write(str(output_path), audio, self.sample_rate)

    def get_info(self) -> Dict[str, Any]:
        """Return model information."""
        return {
            "loaded": self._loaded,
            "device": self.device,
            "speaker_id": self.speaker_id,
            "sample_rate": self.sample_rate,
        }
