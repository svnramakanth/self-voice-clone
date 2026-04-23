from pathlib import Path
import subprocess


class AutoTranscriptionService:
    def transcribe(self, audio_path: str) -> dict:
        path = Path(audio_path)
        if not path.exists():
            raise ValueError("Audio file not found for transcription")

        return {
            "provider": "fallback",
            "text": self._fallback_text(path),
            "confidence": 0.35,
            "segments": self._estimate_segments(path),
            "notes": [
                "No real ASR backend is installed yet. This fallback transcript is metadata-only and not suitable for premium adaptation.",
            ],
        }

    def _estimate_segments(self, audio_path: Path) -> list[dict]:
        duration_seconds = self._duration_seconds(audio_path)
        if duration_seconds <= 0:
            return []
        segment_count = max(1, int(duration_seconds // 8) or 1)
        segment_length = duration_seconds / segment_count
        return [
            {
                "index": index + 1,
                "start_seconds": round(index * segment_length, 2),
                "end_seconds": round(min(duration_seconds, (index + 1) * segment_length), 2),
                "confidence": 0.35,
            }
            for index in range(segment_count)
        ]

    def _duration_seconds(self, audio_path: Path) -> float:
        try:
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(audio_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if probe.stdout.strip():
                return float(probe.stdout.strip())
        except Exception:
            pass
        return 0.0

    def _fallback_text(self, audio_path: Path) -> str:
        try:
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(audio_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            duration = probe.stdout.strip()
            if duration:
                seconds = max(1, int(float(duration)))
                return (
                    "[auto-generated transcript placeholder] "
                    f"Uploaded audio detected with approximate duration {seconds} seconds. "
                    "Install and integrate a real ASR engine like Whisper for accurate transcription."
                )
        except Exception:
            pass

        return (
            "[auto-generated transcript placeholder] Audio was uploaded successfully, "
            "but no real transcription engine is installed yet."
        )