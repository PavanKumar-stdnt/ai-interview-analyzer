"""
src/services/gemini_service.py
────────────────────────────────
Gemini 2.5 Flash integration with enforced structured JSON output.

Uses the modern `google-genai` SDK's response_schema feature to guarantee
a valid InterviewEvaluation object is returned on every call.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Output schema (drives both Pydantic validation & Gemini schema) ─────────────

class TopicScore(BaseModel):
    topic: str
    score: int = Field(ge=0, le=10, description="Score out of 10")
    feedback: str


class InterviewEvaluation(BaseModel):
    overall_score: int = Field(ge=0, le=100, description="Composite score out of 100")
    technical_accuracy: TopicScore
    communication_clarity: TopicScore
    confidence_level: TopicScore
    structure_and_depth: TopicScore
    strengths: list[str] = Field(min_length=1, max_length=5)
    improvement_areas: list[str] = Field(min_length=1, max_length=5)
    recommended_topics: list[str] = Field(
        description="Topics the candidate should study", min_length=1, max_length=4
    )
    tone_summary: str = Field(description="2-3 sentence narrative tone analysis")
    hiring_recommendation: str = Field(
        description="One of: 'Strong Yes', 'Yes', 'Maybe', 'No'"
    )


# ── Service ────────────────────────────────────────────────────────────────────

class GeminiEvaluationService:
    """
    Sends assembled interview context to Gemini 2.5 Flash and parses the
    structured JSON response into a validated InterviewEvaluation instance.
    """

    def __init__(self) -> None:
        from config.settings import settings
        self._settings = settings

    def _build_prompt(
        self,
        transcript: str,
        rubric: dict[str, Any],
        acoustics: dict[str, Any],
        speaker_count: int,
    ) -> str:
        return f"""
You are a senior technical interview coach evaluating a candidate's mock interview performance.

## Interview Transcript
{transcript}

## Reference Rubric
Question: {rubric.get('question', 'N/A')}
Ideal Answer: {rubric.get('ideal_answer', 'N/A')}
Key Concepts Expected: {', '.join(rubric.get('key_concepts', []))}

## Acoustic / Delivery Signals
- Volume: {acoustics.get('volume_label', 'N/A')}
- Pitch Confidence: {acoustics.get('pitch_label', 'N/A')}
- Speech Rate (WPM): {acoustics.get('speech_rate_wpm', 'N/A')} → Pacing: {acoustics.get('pacing_label', 'N/A')}
- Filler Word Count: {acoustics.get('filler_count', 0)}
- Detected Speakers: {speaker_count}

## Task
Evaluate the candidate holistically — technical accuracy, communication clarity, confidence,
and answer structure. Produce a JSON evaluation matching the requested schema exactly.
Scores must reflect rigorous professional standards. Be constructive but honest.
""".strip()

    def evaluate(
        self,
        transcript: str,
        rubric: dict[str, Any],
        acoustics: dict[str, Any],
        speaker_count: int = 1,
    ) -> InterviewEvaluation:
        """
        Call Gemini 2.5 Flash with structured output and return a validated
        InterviewEvaluation.

        Parameters
        ----------
        transcript : str   Full interview transcript.
        rubric     : dict  Matched rubric payload from Qdrant.
        acoustics  : dict  AcousticsResult serialised to dict.
        speaker_count : int  Number of detected speakers.

        Returns
        -------
        InterviewEvaluation
        """
        import google.genai as genai
        import google.genai.types as genai_types

        client = genai.Client(api_key=self._settings.gemini_api_key)
        prompt = self._build_prompt(transcript, rubric, acoustics, speaker_count)

        logger.info("[Gemini] Sending evaluation request to %s …", self._settings.gemini_model)

        try:
            response = client.models.generate_content(
                model=self._settings.gemini_model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=InterviewEvaluation,
                    temperature=0.3,
                ),
            )

            raw_text: str = response.text
            logger.info("[Gemini] Received structured response (%d chars).", len(raw_text))

            evaluation = InterviewEvaluation.model_validate_json(raw_text)
            return evaluation

        except Exception as exc:
            logger.error("[Gemini] Evaluation failed: %s", exc, exc_info=True)
            # Return a safe fallback so the API never crashes on LLM error
            return InterviewEvaluation(
                overall_score=0,
                technical_accuracy=TopicScore(topic="Technical Accuracy", score=0, feedback="Evaluation unavailable."),
                communication_clarity=TopicScore(topic="Communication", score=0, feedback="Evaluation unavailable."),
                confidence_level=TopicScore(topic="Confidence", score=0, feedback="Evaluation unavailable."),
                structure_and_depth=TopicScore(topic="Structure", score=0, feedback="Evaluation unavailable."),
                strengths=["Unable to evaluate at this time."],
                improvement_areas=["Please retry."],
                recommended_topics=["Retry evaluation"],
                tone_summary=f"Evaluation failed due to: {exc}",
                hiring_recommendation="Maybe",
            )
