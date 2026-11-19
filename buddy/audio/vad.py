"""VAD wrapper around pysilero-vad."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    """Voice activity detector using Silero VAD."""

    FRAME_SAMPLES = 512  # 32ms at 16kHz
    FRAME_BYTES = FRAME_SAMPLES * 2  # PCM16 = 2 bytes per sample

    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self._detector = None

    def _ensure_detector(self) -> None:
        if self._detector is None:
            from pysilero_vad import SileroVoiceActivityDetector
            self._detector = SileroVoiceActivityDetector()

    def __call__(self, audio_bytes: bytes) -> float:
        """Return speech probability (0-1) for a 512-sample frame."""
        self._ensure_detector()
        assert self._detector is not None
        return self._detector(audio_bytes)

    def is_speech(self, audio_bytes: bytes) -> bool:
        """Return True if speech probability exceeds threshold."""
        return self(audio_bytes) > self.threshold
