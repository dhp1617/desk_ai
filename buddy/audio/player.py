"""Audio playback with jitter buffer."""
from __future__ import annotations

import asyncio
import collections
import logging
import struct
import threading
import math

logger = logging.getLogger(__name__)


class AudioPlayer:
    """Plays PCM16 audio with a jitter buffer."""

    def __init__(self, sample_rate: int = 24000, jitter_ms: int = 200):
        self.sample_rate = sample_rate
        self.jitter_ms = jitter_ms
        self._stream = None
        self._buffer: collections.deque[int] = collections.deque()
        self._lock = threading.Lock()
        self._running = False
        self._buffering = True  # True = collecting jitter buffer before playing
        self._jitter_samples = int(sample_rate * jitter_ms / 1000)
        self._amplitude: float = 0.0
        self._playing = False

    async def start(self) -> None:
        """Start the output stream."""
        import sounddevice as sd

        def callback(outdata, frames, time_info, status):
            if status:
                logger.warning("Audio output status: %s", status)

            with self._lock:
                if self._buffering and len(self._buffer) < self._jitter_samples:
                    # Still buffering — output silence
                    for i in range(frames):
                        outdata[i * 2:i * 2 + 2] = b"\x00\x00"
                    self._playing = False
                    return

                self._buffering = False
                n = min(frames, len(self._buffer))
                rms_sum = 0.0

                for i in range(n):
                    sample = self._buffer.popleft()
                    outdata[i * 2:i * 2 + 2] = struct.pack("<h", sample)
                    rms_sum += sample * sample

                # Zero-fill remainder
                for i in range(n, frames):
                    outdata[i * 2:i * 2 + 2] = b"\x00\x00"

                self._playing = n > 0

                if n > 0:
                    self._amplitude = min(1.0, math.sqrt(rms_sum / n) / 16384.0)
                else:
                    self._amplitude = 0.0

                # If buffer emptied, go back to buffering mode
                if len(self._buffer) == 0:
                    self._buffering = True

        self._stream = sd.RawOutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=1024,
            callback=callback,
        )
        self._stream.start()
        self._running = True
        logger.info("Audio player started (%dHz)", self.sample_rate)

    async def stop(self) -> None:
        """Stop the output stream."""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def feed(self, audio_bytes: bytes) -> None:
        """Add PCM16 audio to the jitter buffer."""
        if len(audio_bytes) % 2 != 0:
            audio_bytes = audio_bytes[:len(audio_bytes) - 1]
        n_samples = len(audio_bytes) // 2
        samples = struct.unpack(f"<{n_samples}h", audio_bytes)
        with self._lock:
            self._buffer.extend(samples)

    def clear(self) -> None:
        """Clear the buffer immediately (for barge-in)."""
        with self._lock:
            self._buffer.clear()
            self._buffering = True
            self._playing = False
            self._amplitude = 0.0

    def has_buffered_audio(self) -> bool:
        """Check if there is audio queued or currently playing."""
        if not self._running:
            return False
        with self._lock:
            return len(self._buffer) > 0 or self._playing

    async def wait_until_done(self, timeout: float = 30.0) -> None:
        """Wait until all queued audio has finished playing through the speakers."""
        import time
        start = time.monotonic()
        await asyncio.sleep(0.05)
        while self.has_buffered_audio():
            if time.monotonic() - start > timeout:
                break
            await asyncio.sleep(0.04)
        await asyncio.sleep(0.05)

    def is_playing(self) -> bool:
        """Check if audio is currently being played."""
        return self._playing

    def amplitude(self) -> float:
        """Current output amplitude (0-1) for orb visualization."""
        return self._amplitude
