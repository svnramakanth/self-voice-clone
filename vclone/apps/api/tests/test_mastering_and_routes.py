import unittest

from app.api.v1.routes.synthesis import _audio_media_type_for_path
from app.services.mastering import AudioMasteringService


class MasteringAndRouteTests(unittest.TestCase):
    def test_parse_float_accepts_negative_lufs_and_true_peak_values(self):
        service = AudioMasteringService()
        self.assertEqual(service._parse_float("-23.0"), -23.0)
        self.assertEqual(service._parse_float("-16.5"), -16.5)
        self.assertEqual(service._parse_float("-1.0"), -1.0)
        self.assertEqual(service._parse_float("-0.5"), -0.5)

    def test_download_media_types_follow_extension(self):
        self.assertEqual(_audio_media_type_for_path("output.wav"), "audio/wav")
        self.assertEqual(_audio_media_type_for_path("output.flac"), "audio/flac")


if __name__ == "__main__":
    unittest.main()