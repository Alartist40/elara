"""
Piper TTS integration - Fast, local neural TTS
Replaces broken Mimi implementation
"""

import subprocess
import shutil
import numpy as np
import tempfile
import os
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class PiperTTS:
    """
    Piper TTS wrapper. Requires piper-tts binary and voice model.
    Download voices from: https://huggingface.co/rhasspy/piper-voices/tree/main
    """

    SAMPLE_RATE = 22050
    FRAME_SIZE = 1024

    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        speaker_id: Optional[int] = None
    ):
        self.model_path = model_path or os.getenv("ELARA_PIPER_MODEL", "models/piper/en_US-lessac-medium.onnx")
        self.config_path = config_path or os.getenv("ELARA_PIPER_CONFIG", "models/piper/en_US-lessac-medium.onnx.json")
        self.speaker_id = speaker_id
        self.piper_available = self._check_piper()

    def _check_piper(self) -> bool:
        """Check if piper binary exists."""
        return shutil.which("piper") is not None

    def synthesize(
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0,
        chunk_callback: Optional[callable] = None
    ) -> np.ndarray:
        """Synthesize text to speech."""
        if not self.piper_available:
            raise RuntimeError("Piper TTS not available")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Piper model not found: {self.model_path}")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_wav = f.name

        try:
            cmd = [
                "piper",
                "--model", self.model_path,
                "--output_file", temp_wav
            ]

            if self.config_path and os.path.exists(self.config_path):
                cmd.extend(["--config", self.config_path])

            if self.speaker_id is not None:
                cmd.extend(["--speaker", str(self.speaker_id)])

            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            stdout, stderr = process.communicate(input=text, timeout=30)

            if process.returncode != 0:
                raise RuntimeError(f"Piper failed: {stderr}")

            import soundfile as sf
            audio, sr = sf.read(temp_wav)

            if sr != self.SAMPLE_RATE:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.SAMPLE_RATE)

            if speed != 1.0:
                import librosa
                audio = librosa.effects.time_stretch(audio, rate=1.0 / speed)

            if chunk_callback:
                for i in range(0, len(audio), self.FRAME_SIZE):
                    chunk_callback(audio[i:i + self.FRAME_SIZE])

            return audio.astype(np.float32)

        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

    def load_voice(self, name: str, audio_path: str):
        """Stub for compatibility."""
        logger.info(f"Piper uses pre-trained voices. Voice '{name}' registered.")


class StreamingPiperTTS(PiperTTS):
    """Piper TTS with async streaming support."""

    async def synthesize_streaming(self, text: str, voice: str = "default", speed: float = 1.0):
        import asyncio
        loop = asyncio.get_running_loop()
        chunks = []

        def collect_chunk(chunk):
            chunks.append(chunk)

        await loop.run_in_executor(
            None,
            lambda: self.synthesize(text, voice, speed, chunk_callback=collect_chunk)
        )

        for chunk in chunks:
            yield chunk
            await asyncio.sleep(0.01)
