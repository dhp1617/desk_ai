"""Vosk wake word detector using restricted grammar."""
from __future__ import annotations

import json
import logging
import os
import urllib.request
import zipfile
from pathlib import Path

from buddy.wake.base import WakeDetector

logger = logging.getLogger(__name__)

VOSK_MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"


class VoskWakeDetector(WakeDetector):
    """Vosk-based wake word detector restricted to grammar ['{wake_word}', '[unk]']."""

    def __init__(self, wake_word: str = "buddy", model_path: str | Path | None = None):
        self.wake_word = wake_word.strip().lower()
        if model_path:
            self.model_path = Path(model_path)
        else:
            data_dir = Path(os.environ.get("BUDDY_DATA_DIR", Path.home() / ".local" / "share" / "buddy"))
            self.model_path = data_dir / "vosk-model-small-en-us-0.15"

        self._recognizer = None
        self._model = None
        self._initialized = False

    def _ensure_model(self) -> None:
        """Download and unpack model if absent."""
        if self.model_path.is_dir():
            return

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        zip_path = self.model_path.parent / "vosk-model.zip"

        logger.info("Downloading Vosk model to %s...", zip_path)
        urllib.request.urlretrieve(VOSK_MODEL_URL, zip_path)

        logger.info("Extracting Vosk model...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(self.model_path.parent)

        if zip_path.exists():
            zip_path.unlink()

    def _init_engine(self) -> None:
        if self._initialized:
            return

        self._ensure_model()

        # Lazy import of vosk
        try:
            import vosk
        except ImportError:
            raise ImportError(
                "vosk is required for wake word detection. "
                "Install with: pip install vosk"
            )

        # Set vosk log level to silent
        vosk.SetLogLevel(-1)

        self._model = vosk.Model(str(self.model_path))
        grammar = json.dumps([self.wake_word, "[unk]"])
        self._recognizer = vosk.KaldiRecognizer(self._model, 16000.0, grammar)
        self._recognizer.SetWords(True)
        self._initialized = True
        logger.info("Vosk wake detector initialized for wake word '%s'", self.wake_word)

    def feed(self, audio_bytes: bytes) -> bool:
        """Feed PCM16 16kHz audio chunk and detect wake word."""
        self._init_engine()
        assert self._recognizer is not None

        if self._recognizer.AcceptWaveform(audio_bytes):
            res = json.loads(self._recognizer.Result())
            text = res.get("text", "").lower()
            if self.wake_word in text.split():
                self.reset()
                return True
        else:
            partial = json.loads(self._recognizer.PartialResult())
            p_text = partial.get("partial", "").lower()
            if self.wake_word in p_text.split():
                self.reset()
                return True

        return False

    def reset(self) -> None:
        if self._recognizer is not None and self._model is not None:
            import vosk
            grammar = json.dumps([self.wake_word, "[unk]"])
            self._recognizer = vosk.KaldiRecognizer(self._model, 16000.0, grammar)
            self._recognizer.SetWords(True)
