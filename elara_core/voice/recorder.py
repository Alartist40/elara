"""
Asynchronous audio recorder for live voice input.
"""

import asyncio
import numpy as np
import logging
from typing import AsyncGenerator, Optional

logger = logging.getLogger(__name__)

class VoiceRecorder:
    """
    Captures audio from the microphone and yields chunks.
    Requires sounddevice and numpy.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        frame_size: int = 1024,
        device: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_size = frame_size
        self.device = device
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self._is_recording = False

    def _callback(self, indata, frames, time, status):
        """Callback for sounddevice input stream."""
        if status:
            logger.warning(f"Recorder status: {status}")

        # Put a copy of the audio data into the queue
        self._loop.call_soon_threadsafe(self._queue.put_nowait, indata.copy())

    async def stream(self) -> AsyncGenerator[np.ndarray, None]:
        """
        Starts the microphone stream and yields audio chunks.
        """
        import sounddevice as sd

        self._is_recording = True
        self._loop = asyncio.get_running_loop()

        # Ensure we're in a thread-safe way to use sounddevice with asyncio
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self._callback,
            blocksize=self.frame_size,
            device=self.device,
            dtype='float32'
        )

        with stream:
            while self._is_recording:
                chunk = await self._queue.get()
                if not self._is_recording:
                    break
                yield chunk.flatten()

    def stop(self):
        """Stops the recording."""
        self._is_recording = False
        # Feed an empty chunk to unblock the queue.get() if needed
        loop = getattr(self, '_loop', None)
        if loop is not None:
            try:
                loop.call_soon_threadsafe(
                    self._queue.put_nowait,
                    np.zeros((self.frame_size, self.channels), dtype='float32')
                )
            except (RuntimeError, asyncio.QueueFull):
                # Loop might be closed or queue full
                pass
