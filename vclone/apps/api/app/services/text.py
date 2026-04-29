import re

from app.services.pronunciation import PronunciationService


_pronunciation = PronunciationService()


def normalize_text(text: str) -> str:
    normalized = (text or "").strip()
    normalized = normalized.replace("&", " and ")
    normalized = normalized.replace("@", " at ")
    normalized = normalized.replace("%", " percent")
    normalized = normalized.replace("“", '"').replace("”", '"').replace("’", "'")
    normalized = normalized.replace("—", "-").replace("–", "-")
    normalized = re.sub(r"\b(\d+)km\b", r"\1 kilometers", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b(\d+)kg\b", r"\1 kilograms", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
    normalized = re.sub(r"([,.;:!?])(\S)", r"\1 \2", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized, _ = _pronunciation.normalize_for_speech(normalized)
    return normalized.strip()


def chunk_text(text: str, max_chars: int = 220) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    sentence_candidates = [segment.strip() for segment in re.split(r"(?<=[.!?;:])\s+", normalized) if segment.strip()]
    chunks: list[str] = []
    current = ""

    for sentence in sentence_candidates:
        for unit in _split_oversized_sentence(sentence, max_chars=max_chars):
            candidate = f"{current} {unit}".strip() if current else unit
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = unit

    if current:
        chunks.append(current)
    return chunks


def chunk_text_for_clone(text: str, *, mode: str, max_chars: int = 140) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    target_word_limit = 14 if (mode or "preview").lower() == "preview" else 18
    min_word_limit = 8 if (mode or "preview").lower() == "preview" else 10
    sentence_candidates = [segment.strip() for segment in re.split(r"(?<=[.!?;:])\s+", normalized) if segment.strip()]
    chunks: list[str] = []
    current_words: list[str] = []

    for sentence in sentence_candidates:
        words = sentence.split()
        for word in words:
            candidate_words = current_words + [word]
            candidate = " ".join(candidate_words)
            if len(candidate_words) <= target_word_limit and len(candidate) <= max_chars:
                current_words = candidate_words
                continue

            if current_words:
                chunks.append(" ".join(current_words))
            current_words = [word]

        if current_words and len(current_words) >= min_word_limit:
            chunks.append(" ".join(current_words))
            current_words = []

    if current_words:
        if chunks and len(current_words) < min_word_limit:
            merged = f"{chunks[-1]} {' '.join(current_words)}".strip()
            if len(merged) <= max_chars * 1.2:
                chunks[-1] = merged
            else:
                chunks.append(" ".join(current_words))
        else:
            chunks.append(" ".join(current_words))
    return chunks


def split_for_regeneration(text: str, max_chars: int = 140) -> list[str]:
    return chunk_text(text, max_chars=max_chars)


def _split_oversized_sentence(sentence: str, *, max_chars: int) -> list[str]:
    if len(sentence) <= max_chars:
        return [sentence]

    clauses = [segment.strip() for segment in re.split(r"(?<=[,])\s+", sentence) if segment.strip()]
    if len(clauses) > 1:
        grouped: list[str] = []
        current = ""
        for clause in clauses:
            candidate = f"{current} {clause}".strip() if current else clause
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    grouped.append(current)
                current = clause
        if current:
            grouped.append(current)
        if all(len(group) <= max_chars for group in grouped):
            return grouped

    words = sentence.split()
    grouped_words: list[str] = []
    current_words: list[str] = []
    for word in words:
        candidate = " ".join(current_words + [word]).strip()
        if len(candidate) > max_chars and current_words:
            grouped_words.append(" ".join(current_words))
            current_words = [word]
        else:
            current_words.append(word)

    if current_words:
        grouped_words.append(" ".join(current_words))
    return grouped_words
