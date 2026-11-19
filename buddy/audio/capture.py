"""Microphone capture with ring buffer for pre-roll."""
from __future__ import annotations

import asyncio
import collections
import logging
from typing import AsyncIterator

logger = logging.getLogger(__name__)


class AudioCapture:
    """Captures audio from the microphone with a ring buffer for pre-roll."""

    def __init__(self, sample_rate: int = 16000, frame_samples: int = 512):
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.frame_bytes = frame_samples * 2  # PCM16
        self._stream = None
        self._queue: asyncio.Queue[bytes] | None = None
        self._ring: collections.deque[bytes] = collections.deque()
        self._ring_max_frames = 0
        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self, pre_roll_ms: int = 1500) -> None:
        """Start the microphone stream."""
        import sounddevice as sd

        self._loop = asyncio.get_event_loop()
        self._queue = asyncio.Queue()

        # Ring buffer size: enough frames for pre_roll_ms
        frame_ms = (self.frame_samples / self.sample_rate) * 1000
        self._ring_max_frames = max(1, int(pre_roll_ms / frame_ms) + 1)
        self._ring.clear()

        def callback(indata, frames, time_info, status):
            if status:
                logger.warning("Audio input status: %s", status)
            audio_bytes = bytes(indata)
            # Add to ring buffer
            if len(self._ring) >= self._ring_max_frames:
                self._ring.popleft()
            self._ring.append(audio_bytes)
            # Put in queue for async consumers
            if self._queue is not None and self._loop is not None:
                self._loop.call_soon_threadsafe(self._queue.put_nowait, audio_bytes)

        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.frame_samples,
            callback=callback,
        )
        self._stream.start()
        self._running = True
        logger.info("Audio capture started (%dHz, %d samples/frame)",
                     self.sample_rate, self.frame_samples)

    async def stop(self) -> None:
        """Stop the microphone stream."""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        logger.info("Audio capture stopped")

    def flush(self) -> None:
        """Clear all pending frames in queue and ring buffer (e.g. after speaking)."""
        if self._queue is not None:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except Exception:
                    break
        self._ring.clear()

    def get_pre_roll(self, ms: int) -> bytes:
        """Return the last N milliseconds of audio from the ring buffer."""
        frame_ms = (self.frame_samples / self.sample_rate) * 1000
        n_frames = max(1, int(ms / frame_ms))
        frames = list(self._ring)[-n_frames:]
        return b"".join(frames)

    async def frames(self) -> AsyncIterator[bytes]:
        """Async iterator yielding audio frames."""
        if self._queue is None:
            raise RuntimeError("Capture not started")
        while self._running:
            try:
                frame = await asyncio.wait_for(self._queue.get(), timeout=0.2)
                yield frame
            except asyncio.TimeoutError:
                if not self._running:
                    break
                continue

    @property
    def running(self) -> bool:
        return self._running
