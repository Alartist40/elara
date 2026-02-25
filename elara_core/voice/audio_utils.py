"""
Audio utility functions for format conversion and streaming.
"""

import io
import numpy as np
from typing import Optional
import warnings


def pcm_to_float(audio: np.ndarray) -> np.ndarray:
    """Convert PCM integer audio to float32 [-1.0, 1.0]."""
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        return audio.astype(np.float32) / 2147483648.0
    elif audio.dtype in (np.float32, np.float64):
        return audio.astype(np.float32)
    else:
        raise ValueError(f"Unsupported audio dtype: {audio.dtype}")


def float_to_pcm16(audio: np.ndarray) -> np.ndarray:
    """Convert float32 audio to PCM16."""
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype(np.int16)


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    """Convert stereo audio to mono by averaging channels."""
    if len(audio.shape) > 1 and audio.shape[1] > 1:
        return audio.mean(axis=1)
    return audio.squeeze()


def resample(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample audio to target sample rate."""
    if orig_sr == target_sr:
        return audio

    try:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        # Simple linear interpolation fallback
        warnings.warn("librosa not available; using linear interpolation for resampling")
        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def audio_to_wav_bytes(
    audio: np.ndarray,
    sample_rate: int = 16000,
) -> bytes:
    """Convert numpy audio array to WAV bytes."""
    try:
        import soundfile as sf
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format="WAV")
        return buffer.getvalue()
    except ImportError:
        # Fallback: manual WAV header construction
        import struct
        pcm = float_to_pcm16(audio)
        data = pcm.tobytes()
        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',
            36 + len(data),
            b'WAVE',
            b'fmt ',
            16,      # Subchunk1Size
            1,       # AudioFormat (PCM)
            1,       # NumChannels (mono)
            sample_rate,
            sample_rate * 2,  # ByteRate
            2,       # BlockAlign
            16,      # BitsPerSample
            b'data',
            len(data),
        )
        return header + data


def wav_bytes_to_audio(
    wav_bytes: bytes,
) -> tuple[np.ndarray, int]:
    """Convert WAV bytes to numpy array and sample rate."""
    try:
        import soundfile as sf
        audio, sr = sf.read(io.BytesIO(wav_bytes))
        return audio.astype(np.float32), sr
    except ImportError:
        raise ImportError("soundfile required: pip install soundfile")


def detect_audio_format(data: bytes) -> Optional[str]:
    """Detect audio format from file header bytes."""
    if data[:4] == b'RIFF':
        return 'wav'
    elif data[:3] == b'ID3' or data[:2] == b'\xff\xfb':
        return 'mp3'
    elif data[:4] == b'fLaC':
        return 'flac'
    elif data[:4] == b'OggS':
        return 'ogg'
    return None
