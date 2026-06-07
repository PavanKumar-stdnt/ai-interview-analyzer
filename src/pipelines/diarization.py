"""
src/pipelines/diarization.py
─────────────────────────────
Isolated Pyannote speaker diarization wrapper.

Design contract:
- GPU loading happens only inside `run()`.
- The model is deleted and VRAM flushed at the end of `run()` or on exception.
- Returns a list of speaker segment dicts: [{start, end, speaker}, …]
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


@dataclass
class DiarizationResult:
    segments: list[dict[str, Any]] = field(default_factory=list)
    speaker_count: int = 0
    error: str | None = None


class DiarizationPipeline:
    """
    Wraps pyannote.audio's speaker-diarization-3.1 pipeline.

    Example
    -------
    >>> result = DiarizationPipeline().run("interview.wav")
    >>> print(result.segments)
    [{"start": 0.0, "end": 4.2, "speaker": "SPEAKER_00"}, …]
    """

    _MODEL_ID = "pyannote/speaker-diarization-3.1"

    def run(self, audio_path: str | Path) -> DiarizationResult:
        """
        Load the diarization model onto GPU, process *audio_path*, then
        immediately delete the model and flush GPU memory.

        Parameters
        ----------
        audio_path : str | Path
            Path to the audio file (wav / mp3 / flac accepted by pyannote).

        Returns
        -------
        DiarizationResult
            Dataclass with a list of {start, end, speaker} segment dicts.
        """
        # Import here to avoid loading torch at module import time.
        import torch
        from pyannote.audio import Pipeline

        from config.settings import settings

        pipeline: Any = None
        try:
            logger.info("[Diarization] Loading model onto CUDA …")
            pipeline = Pipeline.from_pretrained(
                self._MODEL_ID,
                use_auth_token=settings.hf_token,
            )
            #pipeline.to(torch.device("cuda"))
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipeline.to(torch.device(device))

            logger.info("[Diarization] Running diarization on: %s", audio_path)
            diarization = pipeline(str(audio_path))

            segments: list[dict[str, Any]] = []
            speakers: set[str] = set()

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append(
                    {
                        "start": round(turn.start, 3),
                        "end": round(turn.end, 3),
                        "speaker": speaker,
                    }
                )
                speakers.add(speaker)

            logger.info(
                "[Diarization] Found %d speaker(s), %d segment(s).",
                len(speakers),
                len(segments),
            )
            return DiarizationResult(segments=segments, speaker_count=len(speakers))

        except Exception as exc:
            logger.error("[Diarization] Failed: %s", exc, exc_info=True)
            return DiarizationResult(error=str(exc))

        finally:
            # ── Mandatory VRAM flush ──────────────────────────────────────────
            if pipeline is not None:
                del pipeline
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
                logger.info("[Diarization] VRAM flushed.")
            except Exception:
                pass
