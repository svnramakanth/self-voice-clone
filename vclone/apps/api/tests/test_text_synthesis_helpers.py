import unittest
from types import SimpleNamespace

try:
    import numpy as np
except Exception:  # pragma: no cover - optional quality dependency may be absent.
    np = None

from app.services.synthesis import SynthesisService
from app.services.text import chunk_text_for_clone_plan, normalize_text, prepare_text_for_tts


class TextAndSmoothingHelperTests(unittest.TestCase):
    def test_normalize_removes_spaces_and_expands_abbreviations(self):
        text = "For e.g.: this continues .  It works,  i.e. it is clear."
        normalized = normalize_text(text)
        self.assertNotIn("continues .", normalized)
        self.assertIn("for example", normalized.lower())
        self.assertIn("that is", normalized.lower())

    def test_normalize_preserves_paragraph_boundaries_and_sanskrit_like_words(self):
        normalized = normalize_text("Om Namah Shivaya.\n\nDharma continues .")
        self.assertIn("\n\n", normalized)
        self.assertIn("Namah Shivaya", normalized)
        self.assertIn("continues.", normalized)

    def test_sentence_aware_split_keeps_obvious_sentences(self):
        plan = chunk_text_for_clone_plan("Dr. Rao speaks clearly. This is another sentence.", mode="preview", max_chars=120)
        texts = [item["text"] for item in plan]
        combined = " ".join(texts)
        self.assertIn("Dr. Rao speaks clearly.", combined)
        self.assertIn("This is another sentence.", combined)

    def test_prepare_text_rewrites_parenthetical_as_spoken_aside(self):
        prepared = prepare_text_for_tts("Both concepts (existence and non-existence) can co-exist.")
        self.assertNotIn("(", prepared.synthesis_text)
        self.assertNotIn(")", prepared.synthesis_text)
        self.assertIn("namely existence and non-existence", prepared.synthesis_text)
        self.assertTrue(any(item["type"] == "parenthetical_aside" for item in prepared.replacements))

    def test_prepare_text_removes_quote_marks_without_breaking_contractions(self):
        prepared = prepare_text_for_tts('He said, “God can’t be imagined.”')
        self.assertNotIn('"', prepared.synthesis_text)
        self.assertIn("can't", prepared.synthesis_text)
        self.assertIn("God can't be imagined", prepared.synthesis_text)

    def test_prepare_text_transliterates_iast_for_tts(self):
        prepared = prepare_text_for_tts("Kṛṣṇa teaches that saṃsāra is crossed by jñāna and Ātman knowledge.")
        self.assertIn("Krishna", prepared.synthesis_text)
        self.assertIn("samsaara", prepared.synthesis_text)
        self.assertIn("gyaana", prepared.synthesis_text)
        self.assertIn("Aatman", prepared.synthesis_text)
        self.assertTrue(any(item["type"] == "iast_transliteration" for item in prepared.replacements))
        self.assertTrue(any(item["feature"] == "sanskrit_iast" for item in prepared.prosody_plan))
        self.assertTrue(any(item["feature"] == "devotional_explanatory" for item in prepared.prosody_plan))

    def test_prepare_text_marks_aside_quote_and_devotional_guidance(self):
        prepared = prepare_text_for_tts('God explains “the Self (Ātman) is subtle.”')
        features = {item["feature"] for item in prepared.prosody_plan}
        self.assertIn("aside", features)
        self.assertIn("quotation", features)
        self.assertIn("sanskrit_iast", features)
        self.assertIn("devotional_explanatory", features)

    def test_prepare_text_repairs_hyphenated_line_breaks(self):
        prepared = prepare_text_for_tts("The world is non-\nexistent to the same God.")
        self.assertIn("non-existent", prepared.synthesis_text)
        self.assertNotIn("non- existent", prepared.synthesis_text)

    def test_pause_duration_uses_configured_hints(self):
        service = object.__new__(SynthesisService)
        service.settings = SimpleNamespace(
            synthesis_pause_default_ms=90,
            synthesis_pause_clause_ms=180,
            synthesis_pause_sentence_ms=420,
            synthesis_pause_paragraph_ms=700,
        )
        self.assertEqual(service._pause_ms_for_text("hello", join_hint="clause"), 180)
        self.assertEqual(service._pause_ms_for_text("hello", join_hint="sentence"), 420)
        self.assertEqual(service._pause_ms_for_text("hello", join_hint="paragraph"), 700)

    def test_comfort_noise_pause_avoids_exact_hard_zero_silence(self):
        if np is None:
            self.skipTest("numpy is optional and not installed")
        service = object.__new__(SynthesisService)
        pause = service._build_comfort_noise_pause(np=np, sample_rate=24000, channels=1, pause_ms=120, seed=3)
        self.assertGreater(pause.shape[0], 0)
        self.assertGreater(float(np.max(np.abs(pause))), 0.0)


if __name__ == "__main__":
    unittest.main()