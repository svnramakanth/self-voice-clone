import unittest
from types import SimpleNamespace

try:
    import numpy as np
except Exception:  # pragma: no cover - optional quality dependency may be absent.
    np = None

from app.services.synthesis import SynthesisService
from app.services.text import chunk_text_for_clone_plan, normalize_text


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