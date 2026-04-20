import json
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.generated_asset import GeneratedAsset
from app.models.synthesis_job import SynthesisJob
from app.models.voice_profile import VoiceProfile
from app.services.audit import AuditService
from app.services.storage import StorageService
from app.services.text import chunk_text, normalize_text
from app.services.tts_engine import MockTTSEngine


class SynthesisService:
    def __init__(self, db: Session):
        self.db = db
        self.audit = AuditService(db)
        self.storage = StorageService()
        self.engine = MockTTSEngine()
        self.settings = get_settings()

    def create_job(self, voice_profile_id: str, text: str, mode: str, format: str, sample_rate_hz: int, locale: str) -> SynthesisJob:
        profile = self.db.get(VoiceProfile, voice_profile_id)
        if profile is None:
            raise ValueError("Voice profile not found")
        normalized = normalize_text(text)
        chunks = chunk_text(normalized)
        synthesis = self.engine.synthesize(normalized, voice_profile_id, mode)

        job = SynthesisJob(
            user_id=profile.user_id,
            voice_profile_id=voice_profile_id,
            mode=mode,
            status="completed",
            request_json=json.dumps({"text": text, "format": format, "sample_rate_hz": sample_rate_hz, "locale": locale}),
            normalized_text=normalized,
            output_text_chunks=json.dumps(chunks),
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)

        asset = GeneratedAsset(
            synthesis_job_id=job.id,
            format=format,
            sample_rate=sample_rate_hz,
            channels=1,
            duration_ms=synthesis["duration_ms"],
            object_key=f"synthesis/{profile.user_id}/{job.id}/final/output.{format}",
            watermark_info_json=json.dumps({"provenance": "mvp-placeholder", "engine": synthesis["engine"]}),
        )
        output_dir = Path("uploads") / profile.user_id / "generated" / job.id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"output.{format}"
        output_path.write_text(
            "This is a placeholder synthesized output for the text:\n\n"
            f"{normalized}\n\n"
            f"Voice profile: {profile.name}\n"
            "Replace MockTTSEngine with a real engine to generate actual voice audio.\n"
        )
        asset.object_key = str(output_path)
        self.db.add(asset)
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
        }

    def get_download_url(self, job_id: str) -> dict:
        asset = self.db.query(GeneratedAsset).filter(GeneratedAsset.synthesis_job_id == job_id).one_or_none()
        if asset is None:
            raise ValueError("Generated asset not found")
        return {
            "url": self.storage.build_download_url(asset.object_key),
            "expires_in_seconds": self.settings.signed_url_ttl_seconds,
        }
