from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math
import re


@dataclass
class AlignmentResult:
    source: str
    confidence: float
    transcript_text: str
    normalized_text: str
    segment_count: int
    word_count: int
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


class AlignmentService:
    def align(self, *, transcript_text: str, transcript_path: str | None, duration_seconds: float) -> AlignmentResult:
        provided_text = (transcript_text or "").strip()
        file_text = self._load_text(transcript_path)

        chosen_text = provided_text or file_text
        source = "provided_text" if provided_text else "transcript_file" if file_text else "missing"
        notes: list[str] = []

        normalized = self._normalize(chosen_text)
        word_count = len(normalized.split()) if normalized else 0
        estimated_segments = max(1, math.ceil(duration_seconds / 8)) if duration_seconds > 0 else 1

        confidence = 0.15
        if source == "provided_text":
            confidence = 0.82
        elif source == "transcript_file":
            confidence = 0.75
        else:
            notes.append("No transcript supplied; alignment quality is limited.")

        if duration_seconds > 0 and word_count:
            words_per_second = word_count / duration_seconds
            if words_per_second < 1.0 or words_per_second > 4.5:
                confidence = max(0.25, confidence - 0.2)
                notes.append("Transcript length does not closely match expected speaking rate.")

        if not normalized:
            notes.append("Alignment text is empty.")

        return AlignmentResult(
            source=source,
            confidence=round(confidence, 3),
            transcript_text=chosen_text,
            normalized_text=normalized,
            segment_count=estimated_segments,
            word_count=word_count,
            notes=notes,
        )

    def _load_text(self, transcript_path: str | None) -> str:
        if not transcript_path:
            return ""
        path = Path(transcript_path)
        if not path.exists() or not path.is_file():
            return ""
        try:
            return path.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            return ""

    def _normalize(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", (text or "").strip())
        normalized = re.sub(r"<[^>]+>", " ", normalized)
        return re.sub(r"\s+", " ", normalized).strip()