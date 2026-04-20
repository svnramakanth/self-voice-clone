import json
from pathlib import Path
from uuid import uuid4

from sqlalchemy.orm import Session

from app.models.enrollment import Enrollment
from app.models.user import User
from app.models.voice_profile import VoiceProfile
from app.services.audit import AuditService


class VoiceProfileService:
    def __init__(self, db: Session):
        self.db = db
        self.audit = AuditService(db)

    def _ensure_default_user(self) -> User:
        user = self.db.query(User).filter(User.auth_subject == "demo-user").one_or_none()
        if user:
            return user

        user = User(email="demo@vclone.local", auth_subject="demo-user", status="active")
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def create_simple_profile(
        self,
        name: str,
        transcript_text: str,
        audio_filename: str,
        audio_bytes: bytes,
        transcript_filename: str | None = None,
        transcript_bytes: bytes | None = None,
    ) -> VoiceProfile:
        user = self._ensure_default_user()

        if not transcript_text.strip() and not transcript_bytes:
            raise ValueError("Provide transcript text or an SRT/TXT transcript file")

        enrollment = Enrollment(
            user_id=user.id,
            locale="en-IN",
            consent_text_version="simple-local",
            intended_use="personal_tts",
            liveness_phrase="not-required",
            status="validated",
        )
        self.db.add(enrollment)
        self.db.commit()
        self.db.refresh(enrollment)

        base_dir = Path("uploads") / user.id / enrollment.id
        audio_dir = base_dir / "audio"
        transcript_dir = base_dir / "transcript"
        audio_dir.mkdir(parents=True, exist_ok=True)
        transcript_dir.mkdir(parents=True, exist_ok=True)

        safe_audio_name = f"{uuid4()}-{audio_filename or 'sample-audio.bin'}"
        audio_path = audio_dir / safe_audio_name
        audio_path.write_bytes(audio_bytes)

        transcript_path_str = None
        if transcript_bytes and transcript_filename:
            safe_transcript_name = f"{uuid4()}-{transcript_filename}"
            transcript_path = transcript_dir / safe_transcript_name
            transcript_path.write_bytes(transcript_bytes)
            transcript_path_str = str(transcript_path)

        profile = VoiceProfile(
            user_id=user.id,
            enrollment_id=enrollment.id,
            name=name or "My Voice",
            transcript_text=transcript_text,
            sample_audio_path=str(audio_path),
            transcript_path=transcript_path_str,
            status="ready",
            engine_family="mock",
            base_model_version="simple-mvp-v1",
            readiness_report_json=json.dumps(
                {
                    "audio_uploaded": True,
                    "transcript_provided": bool(transcript_text.strip() or transcript_path_str),
                    "storage": "local",
                    "note": "This MVP stores your sample and transcript locally and uses a mock synthesis engine.",
                }
            ),
        )
        self.db.add(profile)
        self.db.commit()
        self.db.refresh(profile)
        return profile

    def create_profile(self, enrollment_id: str, mode: str, engine_preference: str, allow_adaptation: bool) -> tuple[VoiceProfile, str]:
        enrollment = self.db.get(Enrollment, enrollment_id)
        if enrollment is None:
            raise ValueError("Enrollment not found")
        readiness = {
            "consent": True,
            "liveness": False,
            "speaker_verification": False,
            "validation_status": enrollment.status,
            "adaptation_requested": allow_adaptation,
            "mode": mode,
            "engine_preference": engine_preference,
        }
        profile = VoiceProfile(
            user_id=enrollment.user_id,
            enrollment_id=enrollment_id,
            status="ready" if enrollment.status == "validated" else "pending",
            engine_family="mock",
            base_model_version="mvp-v1",
            readiness_report_json=json.dumps(readiness),
        )
        self.db.add(profile)
        self.db.commit()
        self.db.refresh(profile)
        self.audit.log(
            actor_user_id=enrollment.user_id,
            action="voice_profile.created",
            target_type="voice_profile",
            target_id=profile.id,
            payload=readiness,
        )
        return profile, f"job_profile_{profile.id[:8]}"

    def ensure_schema(self) -> None:
        for column, definition in [
            ("name", "ALTER TABLE voice_profiles ADD COLUMN name VARCHAR(120) DEFAULT 'My Voice'"),
            ("transcript_text", "ALTER TABLE voice_profiles ADD COLUMN transcript_text TEXT DEFAULT ''"),
            ("sample_audio_path", "ALTER TABLE voice_profiles ADD COLUMN sample_audio_path VARCHAR(500) DEFAULT ''"),
            ("transcript_path", "ALTER TABLE voice_profiles ADD COLUMN transcript_path VARCHAR(500)"),
        ]:
            try:
                self.db.execute(f"SELECT {column} FROM voice_profiles LIMIT 1")
            except Exception:
                self.db.execute(definition)
                self.db.commit()

    def list_profiles(self) -> list[VoiceProfile]:
        return self.db.query(VoiceProfile).order_by(VoiceProfile.created_at.desc()).all()

    def get_profile(self, voice_profile_id: str) -> VoiceProfile | None:
        return self.db.get(VoiceProfile, voice_profile_id)
