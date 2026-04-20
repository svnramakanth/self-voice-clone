class MockTTSEngine:
    name = "mock"

    def synthesize(self, text: str, voice_profile_id: str, mode: str) -> dict:
        return {
            "engine": self.name,
            "voice_profile_id": voice_profile_id,
            "mode": mode,
            "preview_text": f"[synthetic preview] {text}",
            "duration_ms": max(len(text) * 45, 500),
        }
