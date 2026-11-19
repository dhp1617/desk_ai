"""Abstract base class for wake word detection."""
from __future__ import annotations

from abc import ABC, abstractmethod


class WakeDetector(ABC):
    """Abstract interface for wake word engines."""

    @abstractmethod
    def feed(self, audio_bytes: bytes) -> bool:
        """Feed PCM16 16kHz audio chunk. Returns True if wake word detected."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset internal engine state after trigger or session transition."""
        pass

    async def start(self) -> None:
        """Initialize or start detector resources."""
        pass

    async def stop(self) -> None:
        """Release detector resources."""
        pass
