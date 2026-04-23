from __future__ import annotations

from dataclasses import asdict, dataclass
import shutil
import subprocess

from app.services.speaker_verification import SpeakerVerificationService


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
    def __init__(self) -> None:
        self.ffmpeg_path = shutil.which("ffmpeg")
        self.speaker_verification = SpeakerVerificationService()

    def score(
        self,
        *,
        audio_path: str,
        duration_seconds: float,
        alignment_confidence: float,
        transcript_confidence: float,
        segment_count: int,
        source_audio_path: str | None = None,
    ) -> EnrollmentQualityReport:
        warnings: list[str] = []

        duration_factor = min(max(duration_seconds / 300.0, 0.0), 1.0)
        snr = round(self._estimate_snr_db(audio_path), 2)
        if source_audio_path:
            speaker_match = self.speaker_verification.verify(
                reference_audio_path=source_audio_path,
                candidate_audio_path=audio_path,
            ).similarity_score
        else:
            speaker_match = min(0.9, 0.55 + duration_factor * 0.25)
        speaker_match = round(speaker_match, 3)
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
            warnings=warnings,
        )

    def _estimate_snr_db(self, audio_path: str) -> float:
        if not self.ffmpeg_path:
            return 18.0

        result = subprocess.run(
            [
                self.ffmpeg_path,
                "-hide_banner",
                "-i",
                audio_path,
                "-af",
                "volumedetect",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        payload = (result.stderr or "") + (result.stdout or "")
        mean_volume = self._extract_metric(payload, "mean_volume")
        max_volume = self._extract_metric(payload, "max_volume")
        if mean_volume is None or max_volume is None:
            return 18.0
        crest = abs(max_volume - mean_volume)
        return max(10.0, min(40.0, 12.0 + crest * 2.0))

    def _extract_metric(self, payload: str, key: str) -> float | None:
        marker = f"{key}:"
        for line in payload.splitlines():
            if marker not in line:
                continue
            try:
                value = line.split(marker, 1)[1].strip().split(" ", 1)[0]
                return float(value)
            except Exception:
                return None
        return None