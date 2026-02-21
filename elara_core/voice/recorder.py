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
        """
        Initialize a VoiceRecorder configured to capture microphone audio.
        
        Parameters:
        	sample_rate (int): Sampling rate in hertz for captured audio.
        	channels (int): Number of audio channels (1 = mono, 2 = stereo).
        	frame_size (int): Number of frames per audio block delivered by the input stream.
        	device (Optional[int]): Optional sound device identifier to use; `None` selects the default device.
        
        The constructor also creates an internal asyncio queue (`_queue`) for incoming audio chunks and initializes the recording flag (`_is_recording`) to False.
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_size = frame_size
        self.device = device
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self._is_recording = False

    def _callback(self, indata, frames, time, status):
        """
        Handle incoming audio frames from the sounddevice input stream and enqueue them for async consumption.
        
        Parameters:
            indata (numpy.ndarray): Audio buffer provided by the input stream.
            frames (int): Number of frames in `indata`.
            time: Timestamp information from the stream callback (passed through, not modified).
            status: Callback status object; if truthy, a warning is logged.
        
        Notes:
            Enqueues a copy of `indata` into the recorder's internal asyncio.Queue in a thread-safe manner.
        """
        if status:
            logger.warning(f"Recorder status: {status}")

        # Put a copy of the audio data into the queue
        loop = asyncio.get_event_loop()
        loop.call_soon_threadsafe(self._queue.put_nowait, indata.copy())

    async def stream(self) -> AsyncGenerator[np.ndarray, None]:
        """
        Provide an asynchronous generator that yields flattened audio blocks captured from the microphone until recording is stopped.
        
        The generator runs the input stream configured with the recorder's sample_rate, channels, frame_size, and device, and yields each captured block as a one-dimensional NumPy array.
        
        Returns:
            np.ndarray: 1-D `float32` array containing a single flattened audio block; its length equals `frame_size * channels`. Recording continues until `stop()` is called.
        """
        import sounddevice as sd

        self._is_recording = True

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
                yield chunk.flatten()

    def stop(self):
        """
        Stop recording and unblock any awaiting stream consumer.
        
        Sets the internal recording flag to False and attempts to enqueue a zero-filled
        audio chunk to release any pending awaiters on the internal queue; silently
        ignores the enqueue if the queue is full.
        """
        self._is_recording = False
        # Feed an empty chunk to unblock the queue.get() if needed
        try:
            self._queue.put_nowait(np.zeros((self.frame_size, self.channels), dtype='float32'))
        except asyncio.QueueFull:
            pass