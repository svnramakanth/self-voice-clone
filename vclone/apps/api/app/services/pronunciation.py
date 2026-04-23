from __future__ import annotations

import re


class PronunciationService:
    LEXICON = {
        "spotify": "spot if eye",
        "ramakanth": "raa maa kanth",
        "coqui": "co kee",
        "xtts": "X T T S",
    }

    def normalize_for_speech(self, text: str) -> tuple[str, list[str]]:
        notes: list[str] = []
        normalized = (text or "").strip()

        normalized = re.sub(r"\bUSD\b", "U S dollars", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bAI\b", "A I", normalized)
        normalized = re.sub(r"\bML\b", "M L", normalized)
        normalized = re.sub(r"\bTTS\b", "T T S", normalized)
        normalized = re.sub(r"\bETA\b", "E T A", normalized)

        for word, spoken in self.LEXICON.items():
            updated = re.sub(rf"\b{re.escape(word)}\b", spoken, normalized, flags=re.IGNORECASE)
            if updated != normalized:
                notes.append(f"Applied pronunciation lexicon override for '{word}'.")
                normalized = updated

        normalized = re.sub(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b", r"\1 slash \2 slash \3", normalized)
        normalized = re.sub(r"\b(\d{1,2}):(\d{2})\b", r"\1 \2", normalized)
        normalized = re.sub(r"\bRs\.?\s*(\d+(?:\.\d+)?)\b", r"\1 rupees", normalized, flags=re.IGNORECASE)

        def replace_currency(match: re.Match[str]) -> str:
            notes.append("Expanded currency token for speech clarity.")
            return f"{match.group(1)} rupees"

        normalized = re.sub(r"₹\s*(\d+(?:\.\d+)?)", replace_currency, normalized)

        def replace_number(match: re.Match[str]) -> str:
            value = match.group(0)
            if len(value) <= 2:
                return value
            notes.append("Expanded long numeric token into grouped speech form.")
            return " ".join(value[i : i + 2] for i in range(0, len(value), 2))

        normalized = re.sub(r"\b\d{3,}\b", replace_number, normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized, notes