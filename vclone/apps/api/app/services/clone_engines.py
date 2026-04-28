from __future__ import annotations

from contextlib import contextmanager
import importlib.util
from pathlib import Path
import shutil

from app.core.config import get_settings
from app.services.tts_engine import BaseTTSEngine, XTTSInferenceError


class CloneEngineInferenceError(XTTSInferenceError):
    pass


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


@contextmanager
def _patch_mps_unavailable(force_disable: bool):
    if not force_disable:
        yield
        return
    try:
        import torch

        original = getattr(torch.backends.mps, "is_available", None)
        if original is None:
            yield
            return
        torch.backends.mps.is_available = lambda: False
        yield
    finally:
        try:
            import torch

            if original is not None:
                torch.backends.mps.is_available = original
        except Exception:
            pass


def _resolve_torch_device(configured_device: str, *, allow_mps_auto: bool = True) -> str:
    if configured_device != "auto":
        return configured_device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if allow_mps_auto and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return "cpu"


class VoxCPM2Engine(BaseTTSEngine):
    name = "voxcpm2"
    native_sample_rate_hz = 48000
    native_channels = 1
    quality_tier = "primary_clone"
    supports_native_distribution_master = False
    _model = None

    def __init__(self) -> None:
        self.settings = get_settings()

    def runtime_status(self) -> dict:
        voxcpm_available = _module_available("voxcpm")
        soundfile_available = _module_available("soundfile")
        device = _resolve_torch_device(self.settings.voxcpm_device, allow_mps_auto=False)
        cpu_blocked = device == "cpu" and not self.settings.voxcpm_allow_cpu
        dependencies_ok = voxcpm_available and soundfile_available
        if cpu_blocked:
            reason = "VoxCPM2 resolved to CPU and CPU execution is disabled. Set VOXCPM_ALLOW_CPU=true to run the primary model slowly on CPU."
        elif dependencies_ok:
            reason = "VoxCPM2 runtime is importable."
        else:
            reason = "Install VoxCPM2 dependencies: pip install -e '.[voxcpm]'"
        return {
            "available": dependencies_ok and not cpu_blocked,
            "reason": reason,
            "dependencies": [
                {"name": "voxcpm", "available": voxcpm_available},
                {"name": "soundfile", "available": soundfile_available},
                {"name": "ffmpeg", "available": shutil.which("ffmpeg") is not None},
                {"name": "ffprobe", "available": shutil.which("ffprobe") is not None},
                {"name": "voxcpm_not_cpu_blocked", "available": not cpu_blocked},
            ],
            "model": self.settings.voxcpm_model_name,
            "device": device,
            "cpu_allowed": self.settings.voxcpm_allow_cpu,
            "device_policy": "auto uses CUDA then CPU for VoxCPM2; Apple MPS is used only when VOXCPM_DEVICE=mps because current VoxCPM2 MPS warm-up can crash with MPSGraph shape errors.",
            "performance_warning": "VoxCPM2 is the primary/best model, but CPU inference may take a long time on first run." if device == "cpu" else None,
        }

    def capabilities(self) -> dict:
        capabilities = super().capabilities()
        capabilities.update(
            {
                "supports_preview": True,
                "supports_final": True,
                "usage": "primary_zero_shot_and_ultimate_clone",
                "recommended_for": ["best_free_local_clone", "ultimate_clone_with_prompt_text", "future_lora_adaptation"],
                "requires_profile_artifacts": ["prompt_audio_path_16k", "prompt_text"],
                "license_note": "VoxCPM2 is advertised upstream as Apache-2.0/commercial-ready; verify exact checkpoint before distribution.",
            }
        )
        return capabilities

    def selection_rationale(self, *, mode: str, sample_rate_hz: int, channels: int) -> list[str]:
        rationale = super().selection_rationale(mode=mode, sample_rate_hz=sample_rate_hz, channels=channels)
        rationale.append("VoxCPM2 is selected because it supports prompt-audio + prompt-text ultimate cloning and native 48 kHz output.")
        rationale.append("For best similarity, enrollment must provide a clean exact prompt clip and transcript from the curated dataset.")
        return rationale

    def _load_model(self):
        if self.__class__._model is not None:
            return self.__class__._model
        try:
            from voxcpm import VoxCPM
        except Exception as exc:
            raise CloneEngineInferenceError("VoxCPM2 is not installed. Install optional clone engine dependencies first.") from exc

        selected_device = _resolve_torch_device(self.settings.voxcpm_device, allow_mps_auto=False)
        optimize = bool(self.settings.voxcpm_optimize and selected_device == "cuda")

        try:
            with _patch_mps_unavailable(force_disable=selected_device == "cpu"):
                model = VoxCPM.from_pretrained(
                    self.settings.voxcpm_model_name,
                    load_denoiser=self.settings.voxcpm_load_denoiser,
                    optimize=optimize,
                )
        except TypeError:
            try:
                with _patch_mps_unavailable(force_disable=selected_device == "cpu"):
                    model = VoxCPM.from_pretrained(
                        self.settings.voxcpm_model_name,
                        load_denoiser=self.settings.voxcpm_load_denoiser,
                    )
            except Exception as exc:
                raise CloneEngineInferenceError(f"Failed to initialize VoxCPM2 model: {exc}") from exc
        except Exception as exc:
            raise CloneEngineInferenceError(f"Failed to initialize VoxCPM2 model: {exc}") from exc

        self.__class__._model = model
        return model

    def synthesize(
        self,
        text: str,
        voice_profile_id: str,
        mode: str,
        output_path: str,
        speaker_wav: str,
        language: str | None = None,
        prompt_text: str | None = None,
        voice_profile_report: dict | None = None,
    ) -> dict:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        reference = Path(speaker_wav)
        if not reference.exists():
            raise CloneEngineInferenceError("VoxCPM2 reference prompt audio was not found")
        runtime = self.runtime_status()
        if not runtime.get("available"):
            raise CloneEngineInferenceError(str(runtime.get("reason") or "VoxCPM2 runtime is unavailable."))

        try:
            import soundfile as sf
        except Exception as exc:
            raise CloneEngineInferenceError("soundfile is required to write VoxCPM2 output audio") from exc

        model = self._load_model()
        kwargs = {
            "text": text,
            "cfg_value": self.settings.voxcpm_cfg_value,
            "inference_timesteps": self.settings.voxcpm_inference_timesteps,
        }
        if prompt_text and prompt_text.strip():
            kwargs.update(
                {
                    "prompt_wav_path": str(reference),
                    "prompt_text": prompt_text.strip(),
                    "reference_wav_path": str(reference),
                }
            )
        else:
            kwargs["reference_wav_path"] = str(reference)

        try:
            wav = model.generate(**kwargs)
        except TypeError:
            fallback_kwargs = {"text": text, "reference_wav_path": str(reference)}
            if prompt_text and prompt_text.strip():
                fallback_kwargs = {"text": text, "prompt_wav_path": str(reference), "prompt_text": prompt_text.strip()}
            wav = model.generate(**fallback_kwargs)
        except Exception as exc:
            raise CloneEngineInferenceError(f"VoxCPM2 inference failed: {exc}") from exc

        sample_rate = int(getattr(getattr(model, "tts_model", None), "sample_rate", self.native_sample_rate_hz) or self.native_sample_rate_hz)
        try:
            sf.write(str(output), wav, sample_rate)
        except Exception as exc:
            raise CloneEngineInferenceError(f"Failed to save VoxCPM2 output: {exc}") from exc

        if not output.exists():
            raise CloneEngineInferenceError("VoxCPM2 inference finished without producing an output file")

        return {
            "engine": self.name,
            "engine_capabilities": self.capabilities(),
            "voice_profile_id": voice_profile_id,
            "mode": mode,
            "preview_text": text,
            "output_created": True,
            "device": _resolve_torch_device(self.settings.voxcpm_device, allow_mps_auto=False),
            "language": language,
            "output_path": str(output),
            "reference_wav": str(reference),
            "prompt_text_used": bool(prompt_text and prompt_text.strip()),
            "profile_dataset_status": (voice_profile_report or {}).get("clone_dataset", {}).get("status"),
        }


class ChatterboxEngine(BaseTTSEngine):
    name = "chatterbox"
    native_sample_rate_hz = 24000
    native_channels = 1
    quality_tier = "secondary_clone"
    supports_native_distribution_master = False
    _model = None

    def __init__(self) -> None:
        self.settings = get_settings()

    def runtime_status(self) -> dict:
        chatterbox_available = _module_available("chatterbox")
        original_available = _module_available("chatterbox.tts")
        multilingual_available = _module_available("chatterbox.mtl_tts")
        turbo_available = _module_available("chatterbox.tts_turbo")
        torchaudio_available = _module_available("torchaudio")
        variant = (self.settings.chatterbox_variant or "original").strip().lower()
        if variant == "turbo" and not turbo_available:
            reason = "Chatterbox Turbo API is not present in the installed package. Install a Turbo-capable Chatterbox build or use CHATTERBOX_VARIANT=original."
        elif variant == "multilingual" and not multilingual_available:
            reason = "Chatterbox multilingual API is not present in the installed package. Use CHATTERBOX_VARIANT=original."
        elif not original_available and variant == "original":
            reason = "Chatterbox original API is not present. Reinstall chatterbox-tts."
        elif chatterbox_available and torchaudio_available:
            reason = "Chatterbox runtime is importable."
        else:
            reason = "Install Chatterbox dependencies separately: pip install -e '.[chatterbox]'"
        variant_available = (
            (variant == "turbo" and turbo_available)
            or (variant == "multilingual" and multilingual_available)
            or (variant == "original" and original_available)
        )
        return {
            "available": chatterbox_available and torchaudio_available and variant_available,
            "reason": reason,
            "dependencies": [
                {"name": "chatterbox", "available": chatterbox_available},
                {"name": "chatterbox.tts", "available": original_available},
                {"name": "chatterbox.mtl_tts", "available": multilingual_available},
                {"name": "chatterbox.tts_turbo", "available": turbo_available},
                {"name": "torchaudio", "available": torchaudio_available},
                {"name": "ffmpeg", "available": shutil.which("ffmpeg") is not None},
            ],
            "variant": variant,
            "device": _resolve_torch_device(self.settings.chatterbox_device, allow_mps_auto=True),
        }

    def capabilities(self) -> dict:
        capabilities = super().capabilities()
        capabilities.update(
            {
                "supports_preview": True,
                "supports_final": True,
                "usage": "secondary_zero_shot_prompt_clone",
                "recommended_for": ["engine_bakeoff", "fallback_when_voxcpm2_unavailable"],
                "requires_profile_artifacts": ["prompt_audio_path_16k"],
                "license_note": "Chatterbox repo is MIT; verify exact model card/license before distribution.",
            }
        )
        return capabilities

    def selection_rationale(self, *, mode: str, sample_rate_hz: int, channels: int) -> list[str]:
        rationale = super().selection_rationale(mode=mode, sample_rate_hz=sample_rate_hz, channels=channels)
        rationale.append("Chatterbox is selected as a practical high-quality zero-shot prompt-cloning fallback.")
        return rationale

    def _load_model(self):
        if self.__class__._model is not None:
            return self.__class__._model
        device = _resolve_torch_device(self.settings.chatterbox_device, allow_mps_auto=True)
        variant = (self.settings.chatterbox_variant or "original").strip().lower()
        try:
            if variant == "multilingual":
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS

                model = ChatterboxMultilingualTTS.from_pretrained(device=device)
            elif variant == "turbo":
                from chatterbox.tts_turbo import ChatterboxTurboTTS

                model = ChatterboxTurboTTS.from_pretrained(device=device)
            else:
                from chatterbox.tts import ChatterboxTTS

                model = ChatterboxTTS.from_pretrained(device=device)
        except Exception as exc:
            raise CloneEngineInferenceError(f"Failed to initialize Chatterbox model: {exc}") from exc
        self.__class__._model = model
        return model

    def synthesize(
        self,
        text: str,
        voice_profile_id: str,
        mode: str,
        output_path: str,
        speaker_wav: str,
        language: str | None = None,
        prompt_text: str | None = None,
        voice_profile_report: dict | None = None,
    ) -> dict:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        reference = Path(speaker_wav)
        if not reference.exists():
            raise CloneEngineInferenceError("Chatterbox reference prompt audio was not found")
        runtime = self.runtime_status()
        if not runtime.get("available"):
            raise CloneEngineInferenceError(str(runtime.get("reason") or "Chatterbox runtime is unavailable."))

        try:
            import torchaudio as ta
        except Exception as exc:
            raise CloneEngineInferenceError("torchaudio is required to save Chatterbox output audio") from exc

        model = self._load_model()
        variant = (self.settings.chatterbox_variant or "original").strip().lower()
        try:
            if variant == "multilingual":
                language_id = (language or self.settings.chatterbox_language_id or "en").split("-")[0].lower()
                wav = model.generate(text, audio_prompt_path=str(reference), language_id=language_id)
            else:
                wav = model.generate(text, audio_prompt_path=str(reference))
        except Exception as exc:
            raise CloneEngineInferenceError(f"Chatterbox inference failed: {exc}") from exc

        sample_rate = int(getattr(model, "sr", self.native_sample_rate_hz) or self.native_sample_rate_hz)
        try:
            ta.save(str(output), wav, sample_rate)
        except Exception as exc:
            raise CloneEngineInferenceError(f"Failed to save Chatterbox output: {exc}") from exc

        if not output.exists():
            raise CloneEngineInferenceError("Chatterbox inference finished without producing an output file")

        return {
            "engine": self.name,
            "engine_capabilities": self.capabilities(),
            "voice_profile_id": voice_profile_id,
            "mode": mode,
            "preview_text": text,
            "output_created": True,
            "device": _resolve_torch_device(self.settings.chatterbox_device, allow_mps_auto=True),
            "language": language,
            "output_path": str(output),
            "reference_wav": str(reference),
            "prompt_text_used": False,
            "profile_dataset_status": (voice_profile_report or {}).get("clone_dataset", {}).get("status"),
        }