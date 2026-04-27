from __future__ import annotations

from app.core.config import get_settings
from app.services.tts_engine import BaseTTSEngine, PremiumFinalEngine, XTTSMasteringEngine, XTTSPreviewEngine


class EngineRegistry:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._preview_engine = XTTSPreviewEngine()
        self._final_engine = XTTSMasteringEngine()
        self._premium_final_engine = PremiumFinalEngine()

    def get_engine(self, mode: str) -> BaseTTSEngine:
        normalized_mode = (mode or "final").strip().lower()
        if normalized_mode == "preview":
            return self._preview_engine
        return self._final_engine

    def describe(self) -> dict:
        return {
            "preview": self._preview_engine.capabilities(),
            "final": self._final_engine.capabilities(),
            "premium_final": self._premium_final_engine.capabilities(),
        }

    def summary(self) -> dict:
        preview = self._preview_engine.capabilities()
        final = self._final_engine.capabilities()
        premium = self._premium_final_engine.capabilities()
        return {
            "recommended_preview_engine": preview["name"],
            "recommended_final_engine": premium["name"] if premium["runtime"]["available"] else final["name"],
            "native_distribution_master_available": bool(
                premium["supports_native_distribution_master"] and premium["runtime"]["available"]
            ) or bool(final["supports_native_distribution_master"] and final["runtime"]["available"]),
            "final_delivery_warning": "Current in-repo XTTS engines remain natively mono/24 kHz, so true Spotify-grade native stereo delivery is still not available.",
        }

    def select(self, mode: str, *, sample_rate_hz: int, channels: int) -> dict:
        normalized_mode = (mode or "final").strip().lower()
        if normalized_mode == "preview":
            engine = self._preview_engine
            selection_reason = "preview mode explicitly requested"
        elif self.settings.engine_prefer_premium_final and self.settings.premium_engine_enabled:
            engine = self._premium_final_engine
            selection_reason = "premium final engine preferred and enabled"
        else:
            engine = self._final_engine
            selection_reason = "fallback to strict final XTTS engine"

        warnings = engine.validate_request(mode=mode, sample_rate_hz=sample_rate_hz, channels=channels)
        rationale = engine.selection_rationale(mode=normalized_mode, sample_rate_hz=sample_rate_hz, channels=channels)
        rationale.append(f"Selection reason: {selection_reason}.")
        return {
            "engine": engine,
            "capabilities": engine.capabilities(),
            "warnings": warnings,
            "selection_reason": selection_reason,
            "rationale": rationale,
        }