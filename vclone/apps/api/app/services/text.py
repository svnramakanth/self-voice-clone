import re


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace("&", " and ")


def chunk_text(text: str, max_chars: int = 120) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []

    for word in words:
        candidate = " ".join(current + [word]).strip()
        if len(candidate) > max_chars and current:
            chunks.append(" ".join(current))
            current = [word]
        else:
            current.append(word)

    if current:
        chunks.append(" ".join(current))
    return chunks
