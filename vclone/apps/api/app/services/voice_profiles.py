import json
from pathlib import Path
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models.enrollment import Enrollment
from app.models.user import User
from app.models.voice_profile import VoiceProfile
from app.services.alignment import AlignmentService
from app.services.audio_processing import AudioProcessingService
from app.services.audit import AuditService
from app.services.quality_scoring import QualityScoringService
from app.services.transcription import AutoTranscriptionService


class VoiceProfileService:
    def __init__(self, db: Session):
        self.db = db
        self.audit = AuditService(db)
        self.transcriber = AutoTranscriptionService()
        self.audio_processor = AudioProcessingService()
        self.alignment = AlignmentService()
        self.quality_scoring = QualityScoringService()

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

        processed_audio = self.audio_processor.process_for_conditioning(str(audio_path), str(audio_dir))

        transcript_path_str = None
        if transcript_bytes and transcript_filename:
            safe_transcript_name = f"{uuid4()}-{transcript_filename}"
            transcript_path = transcript_dir / safe_transcript_name
            transcript_path.write_bytes(transcript_bytes)
            transcript_path_str = str(transcript_path)

        resolved_transcript_text = transcript_text.strip()
        transcription = None
        auto_transcribed = False
        if not resolved_transcript_text:
            transcription = self.transcriber.transcribe(str(audio_path))
            if not transcript_path_str:
                resolved_transcript_text = transcription["text"]
                auto_transcribed = True

        alignment = self.alignment.align(
            transcript_text=resolved_transcript_text,
            transcript_path=transcript_path_str,
            duration_seconds=processed_audio.duration_seconds,
        )
        measured_alignment = self.alignment.analyze_audio_alignment(
            audio_path=processed_audio.processed_path,
            transcript_text=resolved_transcript_text,
        )
        transcript_confidence = transcription["confidence"] if transcription else (0.88 if resolved_transcript_text else 0.25)
        quality = self.quality_scoring.score(
            audio_path=str(audio_path),
            duration_seconds=processed_audio.duration_seconds,
            alignment_confidence=max(alignment.confidence, float(measured_alignment.get("confidence", 0.0) or 0.0)),
            transcript_confidence=transcript_confidence,
            segment_count=max(alignment.segment_count, int(measured_alignment.get("segment_count", 0) or 0)),
            source_audio_path=str(audio_path),
        )

        profile = VoiceProfile(
            user_id=user.id,
            enrollment_id=enrollment.id,
            name=name or "My Voice",
            transcript_text=resolved_transcript_text,
            source_audio_path=str(audio_path),
            sample_audio_path=processed_audio.processed_path,
            transcript_path=transcript_path_str,
            status=processed_audio.readiness_status,
            engine_family="xtts_v2_prep",
            base_model_version="conditioning-phase1",
            readiness_report_json=json.dumps(
                {
                    "audio_uploaded": True,
                    "transcript_provided": bool(resolved_transcript_text or transcript_path_str),
                    "transcript_source": "auto" if auto_transcribed else "provided",
                    "reference_audio_seconds": processed_audio.duration_seconds,
                    "warning_level": processed_audio.warning_level,
                    "guidance": processed_audio.guidance,
                    "alignment": alignment.to_dict(),
                    "measured_alignment": measured_alignment,
                    "quality": quality.to_dict(),
                    "transcription": transcription or {
                        "provider": "provided",
                        "confidence": transcript_confidence,
                        "segments": [],
                        "notes": [],
                    },
                    "audio_processing": {
                        "source_audio_path": str(audio_path),
                        "conditioning_audio_path": processed_audio.processed_path,
                        "ffmpeg_used": processed_audio.ffmpeg_used,
                        "silence_trimmed": processed_audio.silence_trimmed,
                        "loudness_normalized": processed_audio.loudness_normalized,
                        "archival_source_preserved": True,
                        "conditioning_derivative_created": processed_audio.processed_path != str(audio_path),
                    },
                    "storage": "local",
                    "dataset": {
                        "multi_file_support": False,
                        "curated_segment_count": alignment.segment_count,
                        "adaptation_candidate": quality.recommended_for_adaptation,
                    },
                    "note": "Enrollment now preserves the uploaded source audio and stores a separate conditioning derivative, but it is still not a full speaker-verification or fine-tuning pipeline.",
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
            ("source_audio_path", "ALTER TABLE voice_profiles ADD COLUMN source_audio_path VARCHAR(500) DEFAULT ''"),
            ("sample_audio_path", "ALTER TABLE voice_profiles ADD COLUMN sample_audio_path VARCHAR(500) DEFAULT ''"),
            ("transcript_path", "ALTER TABLE voice_profiles ADD COLUMN transcript_path VARCHAR(500)"),
        ]:
            try:
                self.db.execute(text(f"SELECT {column} FROM voice_profiles LIMIT 1"))
            except Exception:
                self.db.execute(text(definition))
                self.db.commit()

    def list_profiles(self) -> list[VoiceProfile]:
        return self.db.query(VoiceProfile).order_by(VoiceProfile.created_at.desc()).all()

    def get_profile(self, voice_profile_id: str) -> VoiceProfile | None:
        return self.db.get(VoiceProfile, voice_profile_id)
