"""
WhisperSTT - OpenAI Whisper wrapper for speech-to-text.
Runs locally, no API calls. Supports multiple model sizes.
"""

import torch
import numpy as np
from typing import Union, Optional, Any
from pathlib import Path
import warnings


class WhisperSTT:
    """
    OpenAI Whisper for speech-to-text.
    Runs locally, no API calls.
    """

    AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large"]

    def __init__(self, model_size: str = "base", device: str = "auto"):
        if model_size not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model {model_size} not in {self.AVAILABLE_MODELS}")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_size = model_size
        self.model = None  # Lazy load
        self.last_duration = 0.0

    def _ensure_loaded(self):
        """Lazy load the Whisper model."""
        if self.model is not None:
            return

        try:
            import whisper as whisper_lib
            self.model = whisper_lib.load_model(self.model_size).to(self.device)
            self._whisper = whisper_lib
        except ImportError:
            raise ImportError(
                "Whisper not installed. Install from: "
                "pip install -e ./whisper"
            )

    def transcribe(
        self,
        audio_input: Union[str, bytes, np.ndarray],
        language: str = "en",
        task: str = "transcribe",
    ) -> str:
        """
        Transcribe audio to text.

        Args:
            audio_input: File path, bytes, or numpy array (16kHz float32)
            language: ISO language code, or None for auto-detect
            task: "transcribe" or "translate" to English

        Returns:
            Transcribed text string.
        """
        self._ensure_loaded()

        # Load audio
        if isinstance(audio_input, str):
            audio = self._whisper.load_audio(audio_input)
        elif isinstance(audio_input, bytes):
            audio = self._bytes_to_audio(audio_input)
        elif isinstance(audio_input, np.ndarray):
            audio = audio_input.astype(np.float32)
        else:
            raise TypeError(f"Unsupported audio input type: {type(audio_input)}")

        self.last_duration = len(audio) / 16000  # Whisper uses 16kHz

        # Pad/trim to 30 seconds
        audio = self._whisper.pad_or_trim(audio)

        # Make log-Mel spectrogram
        mel = self._whisper.log_mel_spectrogram(audio).to(self.model.device)

        # Decode
        options = self._whisper.DecodingOptions(
            language=language,
            task=task,
            fp16=(self.device == "cuda"),
        )
        result = self._whisper.decode(self.model, mel, options)

        return result.text

    def transcribe_with_timestamps(
        self, audio_input: Union[str, bytes]
    ) -> list[dict[str, Any]]:
        """Return transcribed segments with timestamps."""
        self._ensure_loaded()

        if isinstance(audio_input, bytes):
            import tempfile, soundfile as sf, io
            audio_np = self._bytes_to_audio(audio_input)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio_np, 16000)
                audio_input = f.name

        result = self.model.transcribe(
            audio_input, verbose=False, word_timestamps=True
        )
        return result["segments"]

    def _bytes_to_audio(self, audio_bytes: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array (16kHz mono float32)."""
        import io

        try:
            import soundfile as sf
            audio, sr = sf.read(io.BytesIO(audio_bytes))
        except ImportError:
            raise ImportError("soundfile required: pip install soundfile")

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resample to 16kHz if needed
        if sr != 16000:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            except ImportError:
                warnings.warn("librosa not available for resampling; assuming 16kHz")

        return audio.astype(np.float32)

    def get_info(self) -> dict[str, Any]:
        """Return model information."""
        return {
            "model_size": self.model_size,
            "device": self.device,
            "loaded": self.model is not None,
            "last_duration_s": self.last_duration,
        }
