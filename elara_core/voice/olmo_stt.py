"""
OLMoASR STT integration - Fully open source, Apache 2.0
Replaces WhisperSTT
"""

import numpy as np
from typing import Union, Optional, Any
import logging
import os

logger = logging.getLogger(__name__)


class OLMoSTT:
    """
    OLMoASR for speech-to-text.
    Fully offline, no API keys, Apache 2.0 license.
    """

    AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large", "large-v2"]

    def __init__(self, model_size: str = "base", device: str = "auto"):
        if model_size not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model {model_size} not in {self.AVAILABLE_MODELS}")

        self.model_size = model_size
        self.device = device
        self.model = None
        self.last_duration = 0.0

    def _ensure_loaded(self):
        """Lazy load OLMoASR model."""
        if self.model is not None:
            return

        try:
            import olmoasr

            logger.info(f"Loading OLMoASR-{self.model_size}...")
            self.model = olmoasr.load_model(self.model_size, inference=True)

            # Get actual device from model
            if hasattr(self.model, 'device'):
                self.device = str(self.model.device)

            logger.info(f"OLMoASR loaded on {self.device}")

        except ImportError:
            raise ImportError(
                "OLMoASR not installed. Run: pip install olmoasr"
            )
        except Exception as e:
            logger.error(f"Failed to load OLMoASR: {e}")
            raise

    def transcribe(
        self,
        audio_input: Union[str, bytes, np.ndarray],
        language: str = "en",
        task: str = "transcribe"
    ) -> str:
        """
        Transcribe audio to text.

        Args:
            audio_input: File path, bytes, or numpy array (16kHz float32)
            language: Ignored (OLMoASR is English-only for now)
            task: Ignored (transcribe only)

        Returns:
            Transcribed text string
        """
        self._ensure_loaded()

        if isinstance(audio_input, np.ndarray):
            import tempfile
            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
                sf.write(temp_path, audio_input, 16000)

            try:
                result = self.model.transcribe(temp_path)
                self.last_duration = len(audio_input) / 16000
            finally:
                os.remove(temp_path)

        elif isinstance(audio_input, bytes):
            import tempfile
            import soundfile as sf
            import io

            audio_np, sr = sf.read(io.BytesIO(audio_input))
            if sr != 16000:
                try:
                    import librosa
                    audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)
                except ImportError:
                    logger.warning("librosa not available for resampling; assuming 16kHz")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
                sf.write(temp_path, audio_np, 16000)

            try:
                result = self.model.transcribe(temp_path)
                self.last_duration = len(audio_np) / 16000
            finally:
                os.remove(temp_path)
        else:
            # File path
            result = self.model.transcribe(audio_input)
            import soundfile as sf
            info = sf.info(audio_input)
            self.last_duration = info.duration

        return result.get("text", "").strip()

    def transcribe_with_timestamps(
        self,
        audio_input: Union[str, bytes]
    ) -> list[dict[str, Any]]:
        """
        Transcribe with segment timestamps.

        Returns:
            List of segment dicts with start, end, text
        """
        self._ensure_loaded()

        if isinstance(audio_input, bytes):
            import tempfile
            import soundfile as sf
            import io

            audio_np, sr = sf.read(io.BytesIO(audio_input))
            if sr != 16000:
                try:
                    import librosa
                    audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)
                except ImportError:
                    logger.warning("librosa not available for resampling; assuming 16kHz")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
                sf.write(temp_path, audio_np, 16000)

            try:
                result = self.model.transcribe(temp_path)
            finally:
                os.remove(temp_path)
        else:
            result = self.model.transcribe(audio_input)

        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "id": seg.get("id", 0),
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "text": seg.get("text", ""),
                "tokens": seg.get("tokens", []),
                "temperature": seg.get("temperature", 0.0),
                "avg_logprob": seg.get("avg_logprob", 0.0),
                "compression_ratio": seg.get("compression_ratio", 0.0),
                "no_speech_prob": seg.get("no_speech_prob", 0.0)
            })

        return segments

    def get_info(self) -> dict[str, Any]:
        """Get model info."""
        return {
            "model_size": self.model_size,
            "device": self.device,
            "loaded": self.model is not None,
            "last_duration_s": self.last_duration,
            "backend": "OLMoASR",
            "license": "Apache-2.0"
        }
