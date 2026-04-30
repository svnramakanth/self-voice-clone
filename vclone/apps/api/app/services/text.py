import re
from typing import Any

from app.services.pronunciation import PronunciationService


_pronunciation = PronunciationService()
_ABBREVIATION_EXPANSIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bfor\s+e\.g\.\s*:?", flags=re.IGNORECASE), "for example "),
    (re.compile(r"(?<!\w)e\.g\.(?!\w)", flags=re.IGNORECASE), "for example"),
    (re.compile(r"(?<!\w)i\.e\.(?!\w)", flags=re.IGNORECASE), "that is"),
]
_PROTECTED_ABBREVIATIONS = {
    "Mr.": "Mr§",
    "Mrs.": "Mrs§",
    "Ms.": "Ms§",
    "Dr.": "Dr§",
    "Prof.": "Prof§",
    "Sr.": "Sr§",
    "Jr.": "Jr§",
    "vs.": "vs§",
    "etc.": "etc§",
    "St.": "St§",
}


def split_paragraphs(text: str) -> list[str]:
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [segment.strip() for segment in re.split(r"\n\s*\n+", raw) if segment.strip()]
    return paragraphs


def normalize_text(text: str) -> str:
    paragraphs = [_normalize_paragraph(paragraph) for paragraph in split_paragraphs(text)]
    normalized = "\n\n".join(paragraph for paragraph in paragraphs if paragraph)
    return normalized.strip()


def chunk_text_plan(text: str, max_chars: int = 220) -> list[dict[str, Any]]:
    return _chunk_plan(text, max_chars=max_chars, target_word_limit=26, min_word_limit=12)


def chunk_text_for_clone_plan(text: str, *, mode: str, max_chars: int = 140) -> list[dict[str, Any]]:
    normalized_mode = (mode or "preview").lower()
    target_word_limit = 14 if normalized_mode == "preview" else 18
    min_word_limit = 8 if normalized_mode == "preview" else 10
    return _chunk_plan(text, max_chars=max_chars, target_word_limit=target_word_limit, min_word_limit=min_word_limit)


def chunk_text(text: str, max_chars: int = 220) -> list[str]:
    return [item["text"] for item in chunk_text_plan(text, max_chars=max_chars)]


def chunk_text_for_clone(text: str, *, mode: str, max_chars: int = 140) -> list[str]:
    return [item["text"] for item in chunk_text_for_clone_plan(text, mode=mode, max_chars=max_chars)]


def split_for_regeneration(text: str, max_chars: int = 140) -> list[str]:
    return [item["text"] for item in _chunk_plan(text, max_chars=max_chars, target_word_limit=12, min_word_limit=6)]


def _normalize_paragraph(text: str) -> str:
    normalized = (text or "").strip()
    normalized = normalized.replace("&", " and ")
    normalized = normalized.replace("@", " at ")
    normalized = normalized.replace("%", " percent")
    normalized = normalized.replace("“", '"').replace("”", '"').replace("’", "'")
    normalized = normalized.replace("—", "-").replace("–", "-")
    for pattern, replacement in _ABBREVIATION_EXPANSIONS:
        normalized = pattern.sub(replacement, normalized)
    normalized = re.sub(r"\b(\d+)km\b", r"\1 kilometers", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b(\d+)kg\b", r"\1 kilograms", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
    normalized = re.sub(r"([,.;:!?]){2,}", lambda match: match.group(0)[0], normalized)
    normalized = re.sub(r":\s*(?=[,.;:!?])", " ", normalized)
    normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
    normalized = re.sub(r"([,.;:!?])(\S)", r"\1 \2", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized, _ = _pronunciation.normalize_for_speech(normalized)
    return normalized.strip()


def _chunk_plan(text: str, *, max_chars: int, target_word_limit: int, min_word_limit: int) -> list[dict[str, Any]]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    units: list[dict[str, str]] = []
    paragraphs = split_paragraphs(normalized)
    for paragraph_index, paragraph in enumerate(paragraphs):
        sentences = _split_sentences(paragraph)
        if not sentences:
            continue
        for sentence_index, sentence in enumerate(sentences):
            is_last_sentence = sentence_index == len(sentences) - 1
            boundary = "paragraph" if is_last_sentence and paragraph_index < len(paragraphs) - 1 else "sentence"
            oversized_units = _split_oversized_sentence(sentence, max_chars=max_chars)
            for oversize_index, unit in enumerate(oversized_units):
                unit_boundary = boundary if oversize_index == len(oversized_units) - 1 else "clause"
                units.append({"text": unit, "break_after": unit_boundary})

    chunks: list[dict[str, Any]] = []
    current_text = ""
    current_word_count = 0
    current_boundary = "soft"

    for unit in units:
        unit_text = str(unit["text"]).strip()
        if not unit_text:
            continue
        unit_word_count = len(unit_text.split())
        candidate = f"{current_text} {unit_text}".strip() if current_text else unit_text
        candidate_word_count = current_word_count + unit_word_count
        if current_text and (len(candidate) > max_chars or candidate_word_count > target_word_limit):
            chunks.append({"text": current_text, "break_after": current_boundary})
            current_text = unit_text
            current_word_count = unit_word_count
            current_boundary = str(unit.get("break_after") or "soft")
            continue
        current_text = candidate
        current_word_count = candidate_word_count
        current_boundary = str(unit.get("break_after") or current_boundary)
        if current_word_count >= min_word_limit and current_boundary in {"sentence", "paragraph"}:
            chunks.append({"text": current_text, "break_after": current_boundary})
            current_text = ""
            current_word_count = 0
            current_boundary = "soft"

    if current_text:
        if chunks and current_word_count < min_word_limit:
            merged = f"{chunks[-1]['text']} {current_text}".strip()
            if len(merged) <= int(max_chars * 1.2):
                chunks[-1]["text"] = merged
                chunks[-1]["break_after"] = current_boundary
            else:
                chunks.append({"text": current_text, "break_after": current_boundary})
        else:
            chunks.append({"text": current_text, "break_after": current_boundary})
    return chunks


def _split_sentences(paragraph: str) -> list[str]:
    protected = paragraph
    for original, token in _PROTECTED_ABBREVIATIONS.items():
        protected = re.sub(re.escape(original), token, protected, flags=re.IGNORECASE)
    protected = re.sub(r"(?<=\d)\.(?=\d)", "§", protected)
    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", protected) if segment.strip()]
    restored: list[str] = []
    for sentence in sentences:
        sentence = sentence.replace("§", ".")
        for original, token in _PROTECTED_ABBREVIATIONS.items():
            sentence = re.sub(re.escape(token), original, sentence, flags=re.IGNORECASE)
        restored.append(sentence.strip())
    return restored


def _split_oversized_sentence(sentence: str, *, max_chars: int) -> list[str]:
    if len(sentence) <= max_chars:
        return [sentence]

    clauses = [segment.strip() for segment in re.split(r"(?<=[,;:])\s+", sentence) if segment.strip()]
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
