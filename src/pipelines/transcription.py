"""
src/pipelines/transcription.py
────────────────────────────────
Isolated Faster-Whisper transcription wrapper.

Design contract:
- Model is instantiated and destroyed inside `run()`.
- VRAM is flushed in the finally block regardless of success/failure.
- Returns a plain string transcript and per-segment timing data.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logger = logging.getLogger(__name__)

#from faster_whisper import WhisperModel

#MODEL = WhisperModel(
 #   "base",
  #  device="cuda",
   # compute_type="int8_float16"
#)

@dataclass
class TranscriptionResult:
    transcript: str = ""
    segments: list[dict[str, Any]] = field(default_factory=list)
    language: str = "en"
    error: str | None = None


class TranscriptionPipeline:
    """
    Wraps faster-whisper's WhisperModel for single-file transcription.

    The model runs on GPU with ``compute_type="int8_float16"`` to respect the
    4 GB VRAM budget of the RTX 3050.

    Example
    -------
    >>> result = TranscriptionPipeline().run("interview.wav")
    >>> print(result.transcript)
    'Good morning, my name is …'
    """

    def run(self, audio_path: str | Path) -> TranscriptionResult:
        """
        Instantiate the Whisper model, transcribe *audio_path*, then delete
        the model and flush GPU memory.

        Parameters
        ----------
        audio_path : str | Path
            Path to the audio file.

        Returns
        -------
        TranscriptionResult
            Dataclass with full transcript string and per-segment list.
        """
        from faster_whisper import WhisperModel

        from config.settings import settings

        model: Any = None
        try:
            logger.info("[Transcription] Loading Whisper '%s' on cpu …", settings.whisper_model_size)
            model = WhisperModel(
                "small",
                device="cpu",
               compute_type="int8",
            )

            logger.info("[Transcription] Transcribing: %s", audio_path)
            raw_segments, info = model.transcribe(
                str(audio_path),
                beam_size=1,
                vad_filter=True,         # remove silent regions
                vad_parameters=dict(min_silence_duration_ms=300),
            )

            segments: list[dict[str, Any]] = []
            full_text_parts: list[str] = []

            for seg in raw_segments:
                segments.append(
                    {
                        "start": round(seg.start, 3),
                        "end": round(seg.end, 3),
                        "text": seg.text.strip(),
                    }
                )
                full_text_parts.append(seg.text.strip())

            transcript = " ".join(full_text_parts)
            logger.info(
                "[Transcription] Done. Language=%s, %d segment(s), %d chars.",
                info.language,
                len(segments),
                len(transcript),
            )
            return TranscriptionResult(
                transcript=transcript,
                segments=segments,
                language=info.language,
            )

        except Exception as exc:
            logger.error("[Transcription] Failed: %s", exc, exc_info=True)
            return TranscriptionResult(error=str(exc))

        finally:
            # ── Mandatory VRAM flush ──────────────────────────────────────────
            if model is not None:
              del model
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
                logger.info("[Transcription] VRAM flushed.")
            except Exception:
                pass
