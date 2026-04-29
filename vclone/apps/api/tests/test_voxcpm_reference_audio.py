import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services.clone_engines import CloneEngineInferenceError, VoxCPM2Engine


class VoxCPM2ReferenceAudioTests(unittest.TestCase):
    def test_prepare_reference_audio_returns_original_when_already_16k_mono(self):
        engine = VoxCPM2Engine()
        with tempfile.TemporaryDirectory() as tmpdir:
            reference = Path(tmpdir) / "reference.wav"
            reference.write_bytes(b"placeholder")

            with patch("app.services.clone_engines._duration_seconds", return_value=4.0):
                with patch.object(engine, "_reference_stream_info", return_value=(16000, 1)):
                    prepared = engine._prepare_reference_audio(reference)

        self.assertEqual(prepared, reference)

    def test_prepare_reference_audio_normalizes_non_16k_reference(self):
        engine = VoxCPM2Engine()
        with tempfile.TemporaryDirectory() as tmpdir:
            reference = Path(tmpdir) / "reference.wav"
            normalized = Path(tmpdir) / "reference-16k.wav"
            reference.write_bytes(b"placeholder")

            with patch("app.services.clone_engines._duration_seconds", return_value=4.0):
                with patch.object(engine, "_reference_stream_info", side_effect=[(48000, 2), (None, None), (16000, 1)]):
                    with patch.object(engine, "_normalized_reference_cache_path", return_value=normalized):
                        with patch.object(engine, "_normalize_reference_audio", side_effect=lambda _src, dst: dst.write_bytes(b"normalized")):
                            prepared = engine._prepare_reference_audio(reference)

        self.assertEqual(prepared, normalized)

    def test_prepare_reference_audio_rejects_too_short_reference(self):
        engine = VoxCPM2Engine()
        with tempfile.TemporaryDirectory() as tmpdir:
            reference = Path(tmpdir) / "reference.wav"
            reference.write_bytes(b"placeholder")

            with patch("app.services.clone_engines.validate_voxcpm_reference_audio") as validation_mock:
                validation_mock.return_value.valid = False
                validation_mock.return_value.message = "reference_audio_too_short: duration=0.166s, min=5.0s"
                with self.assertRaises(CloneEngineInferenceError) as exc:
                    engine._prepare_reference_audio(reference)

        self.assertIn("reference_audio_too_short", str(exc.exception))


if __name__ == "__main__":
    unittest.main()