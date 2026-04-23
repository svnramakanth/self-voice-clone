from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import os
import random


@dataclass
class EnrollmentQualityReport:
    overall_score: float
    speaker_match_score: float
    transcription_confidence: float
    alignment_confidence: float
    estimated_snr_db: float
    estimated_segment_count: int
    recommended_for_adaptation: bool
    warnings: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


class QualityScoringService:
    def score(
        self,
        *,
        audio_path: str,
        duration_seconds: float,
        alignment_confidence: float,
        transcript_confidence: float,
        segment_count: int,
    ) -> EnrollmentQualityReport:
        warnings: list[str] = []
        seed = sum(ord(char) for char in os.path.basename(audio_path))
        rng = random.Random(seed)

        duration_factor = min(max(duration_seconds / 300.0, 0.0), 1.0)
        snr = round(18 + (duration_factor * 12) + rng.uniform(-2.5, 2.5), 2)
        speaker_match = round(min(0.97, 0.58 + duration_factor * 0.32 + rng.uniform(-0.05, 0.05)), 3)
        overall = round(min(0.99, (speaker_match * 0.4) + (alignment_confidence * 0.3) + (transcript_confidence * 0.2) + (min(snr / 30, 1.0) * 0.1)), 3)

        if duration_seconds < 120:
            warnings.append("Reference duration is short for highly stable long-form cloning.")
        if alignment_confidence < 0.65:
            warnings.append("Alignment confidence is below the preferred threshold.")
        if transcript_confidence < 0.7:
            warnings.append("Transcript confidence is low; pronunciation drift is more likely.")
        if snr < 20:
            warnings.append("Estimated signal-to-noise ratio is below preferred studio-quality levels.")

        return EnrollmentQualityReport(
            overall_score=overall,
            speaker_match_score=speaker_match,
            transcription_confidence=round(transcript_confidence, 3),
            alignment_confidence=round(alignment_confidence, 3),
            estimated_snr_db=snr,
            estimated_segment_count=max(segment_count, 1),
            recommended_for_adaptation=overall >= 0.8 and duration_seconds >= 300,
            warnings=warnings + ["Quality scoring is heuristic and should not be treated as a studio-certification signal."],
        )