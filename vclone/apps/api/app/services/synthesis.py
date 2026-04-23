import json
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.generated_asset import GeneratedAsset
from app.models.synthesis_job import SynthesisJob
from app.models.voice_profile import VoiceProfile
from app.services.audit import AuditService
from app.services.asr_backcheck import ASRBackcheckService
from app.services.evaluation import EvaluationService
from app.services.engine_registry import EngineRegistry
from app.services.mastering import AudioMasteringService
from app.services.post_synthesis_qc import PostSynthesisQCService
from app.services.storage import StorageService
from app.services.text import chunk_text, normalize_text, split_for_regeneration
from app.services.tts_engine import XTTSInferenceError


class SynthesisService:
    def __init__(self, db: Session):
        self.db = db
        self.audit = AuditService(db)
        self.storage = StorageService()
        self.engine_registry = EngineRegistry()
        self.mastering = AudioMasteringService()
        self.post_qc = PostSynthesisQCService()
        self.asr_backcheck = ASRBackcheckService()
        self.evaluation = EvaluationService()
        self.settings = get_settings()

    def create_job(
        self,
        voice_profile_id: str,
        text: str,
        mode: str,
        format: str,
        sample_rate_hz: int,
        locale: str,
        channels: int = 2,
        require_native_master: bool = False,
    ) -> SynthesisJob:
        profile = self.db.get(VoiceProfile, voice_profile_id)
        if profile is None:
            raise ValueError("Voice profile not found")
        if profile.status not in {"ready", "ready_with_warning"}:
            raise ValueError("Voice profile is not ready for XTTS inference yet")

        delivery_request = self.mastering.normalize_delivery_request(format, sample_rate_hz, channels)
        engine_selection = self.engine_registry.select(
            mode,
            sample_rate_hz=int(delivery_request["sample_rate_hz"]),
            channels=int(delivery_request["channels"]),
        )
        engine = engine_selection["engine"]
        normalized = normalize_text(text)
        chunks = chunk_text(normalized)
        if not chunks:
            raise ValueError("Text to synthesize is empty after normalization")
        qc_plan = self.post_qc.evaluate_chunks(chunks)
        regeneration_plan = self.post_qc.regeneration_plan(chunks, qc_plan)
        if mode == "final" and (self.settings.fail_on_derived_final_master or require_native_master):
            if int(delivery_request["sample_rate_hz"]) > engine.native_sample_rate_hz or int(delivery_request["channels"]) > engine.native_channels:
                raise ValueError(
                    "Requested final delivery would be derived rather than native. Lower the requested delivery spec, switch to preview/mono 24 kHz, or integrate a native final-render engine."
                )

        job = SynthesisJob(
            user_id=profile.user_id,
            voice_profile_id=voice_profile_id,
            mode=mode,
            status="queued",
            request_json=json.dumps(
                {
                    "text": text,
                    "format": delivery_request["format"],
                    "sample_rate_hz": delivery_request["sample_rate_hz"],
                    "channels": delivery_request["channels"],
                    "locale": locale,
                    "require_native_master": require_native_master,
                    "engine": engine_selection["capabilities"],
                    "engine_registry": self.engine_registry.describe(),
                    "engine_warnings": engine_selection["warnings"],
                    "engine_selection_reason": engine_selection["selection_reason"],
                    "engine_selection_rationale": engine_selection["rationale"],
                    "preflight_qc": qc_plan,
                    "regeneration_plan": regeneration_plan,
                }
            ),
            normalized_text=normalized,
            output_text_chunks=json.dumps(chunks),
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)

        output_dir = Path("uploads") / profile.user_id / "generated" / job.id
        output_dir.mkdir(parents=True, exist_ok=True)
        chunk_dir = output_dir / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        native_mix_path = output_dir / "native-mix.wav"
        output_path = output_dir / f"output.{delivery_request['format']}"

        job.status = "running"
        self.db.add(job)
        self.db.commit()

        try:
            chunk_paths: list[str] = []
            engine_runs: list[dict] = []
            language_code = (locale or self.settings.xtts_default_language).split("-")[0].lower()
            for index, chunk in enumerate(chunks, start=1):
                chunk_path = chunk_dir / f"chunk-{index:03d}.wav"
                render_text = chunk
                if index in qc_plan.get("failed_segments", []):
                    split_version = split_for_regeneration(chunk)
                    render_text = " ".join(split_version)
                synthesis = engine.synthesize(
                    render_text,
                    voice_profile_id,
                    mode,
                    str(chunk_path),
                    speaker_wav=profile.sample_audio_path,
                    language=language_code,
                )
                engine_runs.append(synthesis)
                chunk_paths.append(str(chunk_path))

            concat_info = self.mastering.concatenate_wav_chunks(chunk_paths, str(native_mix_path))
            delivery_report = self.mastering.master_audio(
                str(native_mix_path),
                str(output_path),
                audio_format=str(delivery_request["format"]),
                sample_rate_hz=int(delivery_request["sample_rate_hz"]),
                channels=int(delivery_request["channels"]),
            )
            asr_backcheck = self.asr_backcheck.evaluate(expected_text=normalized, chunks=chunks, audio_path=str(output_path))
            evaluation_report = self.evaluation.evaluate(
                audio_path=str(output_path),
                reference_path=profile.sample_audio_path,
                expected_text=normalized,
                chunks=chunks,
            )
        except XTTSInferenceError as exc:
            job.status = "failed"
            self.db.add(job)
            self.db.commit()
            raise ValueError(str(exc)) from exc
        except ValueError:
            job.status = "failed"
            self.db.add(job)
            self.db.commit()
            raise
        except Exception as exc:
            job.status = "failed"
            self.db.add(job)
            self.db.commit()
            raise ValueError(f"Synthesis failed: {exc}") from exc

        asset = GeneratedAsset(
            synthesis_job_id=job.id,
            format=str(delivery_request["format"]),
            sample_rate=delivery_report["delivery"]["sample_rate_hz"],
            channels=delivery_report["delivery"]["channels"],
            duration_ms=delivery_report["delivery"]["duration_ms"],
            object_key=f"synthesis/{profile.user_id}/{job.id}/final/output.{delivery_request['format']}",
            watermark_info_json=json.dumps(
                {
                    "provenance": "xtts-phase2",
                    "engine": engine_runs[0]["engine"] if engine_runs else engine.name,
                    "device": engine_runs[0].get("device") if engine_runs else None,
                    "language": engine_runs[0].get("language") if engine_runs else None,
                    "chunk_count": len(chunks),
                    "qc": qc_plan,
                    "regeneration_plan": regeneration_plan,
                    "asr_backcheck": asr_backcheck,
                    "evaluation": evaluation_report.to_dict(),
                    "concat": concat_info,
                    "engine_selection": {
                        "requested_mode": mode,
                        "resolved_engine": engine_selection["capabilities"],
                        "selection_reason": engine_selection["selection_reason"],
                        "rationale": engine_selection["rationale"],
                        "warnings": engine_selection["warnings"],
                    },
                    "engine_registry": self.engine_registry.describe(),
                    "delivery_report": delivery_report,
                }
            ),
            checksum_sha256=delivery_report["delivery"]["checksum_sha256"],
        )

        if require_native_master and not delivery_report["spotify"]["native_master_ok"]:
            job.status = "failed"
            self.db.add(job)
            self.db.commit()
            raise ValueError("Native master was required, but the rendered asset did not validate as a native master.")

        if mode == "final" and self.settings.fail_on_derived_final_master and not delivery_report["spotify"]["native_master_ok"]:
            job.status = "failed"
            self.db.add(job)
            self.db.commit()
            raise ValueError("Final delivery validation failed because the produced file is not a true native master.")

        if mode == "final" and engine.supports_native_distribution_master and not delivery_report["spotify"]["native_master_ok"]:
            watermark_payload = json.loads(asset.watermark_info_json)
            watermark_payload["engine_selection"] = {
                "requested_mode": mode,
                "resolved_engine": engine_selection["capabilities"],
                "selection_reason": engine_selection["selection_reason"],
                "rationale": engine_selection["rationale"],
                "warnings": engine_selection["warnings"] + [
                    "Engine advertised native distribution capability, but the produced file was not validated as native."
                ],
            }
            asset.watermark_info_json = json.dumps(watermark_payload)

        asset.object_key = str(output_path)
        self.db.add(asset)
        job.status = "completed"
        self.db.add(job)
        self.db.commit()
        self.audit.log(actor_user_id=profile.user_id, action="synthesis.created", target_type="synthesis_job", target_id=job.id)
        return job

    def get_job(self, job_id: str) -> SynthesisJob | None:
        return self.db.get(SynthesisJob, job_id)

    def get_preview(self, job_id: str) -> dict:
        job = self.db.get(SynthesisJob, job_id)
        if job is None:
            raise ValueError("Synthesis job not found")
        return {
            "job_id": job.id,
            "status": job.status,
            "preview_text": job.normalized_text,
            "chunks": json.loads(job.output_text_chunks),
            "request": json.loads(job.request_json),
        }

    def get_download_url(self, job_id: str) -> dict:
        asset = self.db.query(GeneratedAsset).filter(GeneratedAsset.synthesis_job_id == job_id).one_or_none()
        if asset is None:
            raise ValueError("Generated asset not found")
        metadata = json.loads(asset.watermark_info_json or "{}")
        return {
            "url": self.storage.build_download_url(asset.object_key),
            "expires_in_seconds": self.settings.signed_url_ttl_seconds,
            "asset": {
                "format": asset.format,
                "sample_rate_hz": asset.sample_rate,
                "channels": asset.channels,
                "duration_ms": asset.duration_ms,
                "local_path": asset.object_key,
                "checksum_sha256": asset.checksum_sha256,
            },
            "delivery_report": metadata.get("delivery_report", {}),
            "evaluation": metadata.get("evaluation", {}),
            "asr_backcheck": metadata.get("asr_backcheck", {}),
            "engine_selection": metadata.get("engine_selection", {}),
            "engine_registry": metadata.get("engine_registry", self.engine_registry.describe()),
        }
