from __future__ import annotations

import re
from typing import Any


class ASRBackcheckService:
    def evaluate(self, *, expected_text: str, chunks: list[str]) -> dict[str, Any]:
        normalized_expected = self._normalize(expected_text)
        normalized_render = self._normalize(" ".join(chunks))

        expected_words = normalized_expected.split()
        render_words = normalized_render.split()
        length_gap = abs(len(expected_words) - len(render_words))
        base_wer = 0.02 + (length_gap / max(len(expected_words), 1))
        punctuation_penalty = min(sum(chunk.count(",") for chunk in chunks) * 0.002, 0.08)
        estimated_wer = round(min(0.6, base_wer + punctuation_penalty), 3)
        intelligibility = round(max(0.0, 1.0 - estimated_wer), 3)

        return {
            "provider": "heuristic",
            "is_measured": False,
            "estimated_wer": estimated_wer,
            "intelligibility_score": intelligibility,
            "dependency_hook": "Replace with Whisper/faster-whisper back-check for real WER.",
            "release_blocker": "Do not treat this heuristic WER as release-grade validation.",
        }

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())).strip()