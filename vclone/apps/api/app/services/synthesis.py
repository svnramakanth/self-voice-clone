import json
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import time
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import SessionLocal
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


class SynthesisCancelledError(RuntimeError):
    pass


def _synthesis_engine_worker(payload: dict, result_queue) -> None:
    from app.services.engine_registry import EngineRegistry

    try:
        engine = EngineRegistry().get_engine_by_name(str(payload["engine_name"]))
        chunk_paths: list[str] = []
        engine_runs: list[dict] = []
        chunk_texts: list[str] = list(payload.get("chunk_texts") or [])
        chunk_dir = Path(str(payload["chunk_dir"]))
        chunk_dir.mkdir(parents=True, exist_ok=True)
        for index, render_text in enumerate(chunk_texts, start=1):
            result_queue.put(
                {
                    "type": "progress",
                    "current_chunk": index,
                    "total_chunks": len(chunk_texts),
                }
            )
            chunk_path = chunk_dir / f"chunk-{index:03d}.wav"
            synthesis = engine.synthesize(
                render_text,
                str(payload["voice_profile_id"]),
                str(payload["mode"]),
                str(chunk_path),
                speaker_wav=str(payload["speaker_wav"]),
                language=payload.get("language"),
                prompt_text=payload.get("prompt_text"),
                voice_profile_report=payload.get("voice_profile_report") or {},
            )
            chunk_paths.append(str(chunk_path))
            engine_runs.append(synthesis)
        result_queue.put({"type": "done", "chunk_paths": chunk_paths, "engine_runs": engine_runs})
    except BaseException as exc:  # noqa: BLE001 - worker must report all failures to parent when possible.
        result_queue.put({"type": "error", "error": f"{type(exc).__name__}: {exc}"})


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
            raise ValueError(
                f"Voice profile is not ready for clone inference yet (status={profile.status}). "
                "Re-run enrollment/retry processing so the profile has curated clone prompt artifacts."
            )
        profile_report = self._profile_report(profile)
        reference_path, prompt_text = self._resolve_clone_reference(profile, profile_report)
        if not reference_path or not Path(reference_path).exists():
            raise ValueError("Voice profile clone prompt/reference audio is missing. Re-run enrollment processing from the original upload.")

        delivery_request = self.mastering.normalize_delivery_request(format, sample_rate_hz, channels)
        engine_selection = self.engine_registry.select(
            mode,
            sample_rate_hz=int(delivery_request["sample_rate_hz"]),
            channels=int(delivery_request["channels"]),
        )
        engine = engine_selection["engine"]
        if not engine_selection["capabilities"].get("runtime", {}).get("available"):
            reason = engine_selection["capabilities"].get("runtime", {}).get("reason", "Selected clone engine is unavailable.")
            raise ValueError(f"Selected clone engine '{engine.name}' is unavailable. {reason}")
        normalized = normalize_text(text)
        chunks = chunk_text(normalized, max_chars=self.settings.synthesis_max_chunk_chars)
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
                    "clone_reference_audio_path": reference_path,
                    "clone_prompt_text_available": bool(prompt_text),
                    "clone_dataset_status": profile_report.get("clone_dataset", {}).get("status"),
                    "engine_warnings": engine_selection["warnings"],
                    "engine_selection_reason": engine_selection["selection_reason"],
                    "engine_selection_rationale": engine_selection["rationale"],
                    "preflight_qc": qc_plan,
                    "regeneration_plan": regeneration_plan,
                    "cancel_requested": False,
                    "last_heartbeat_at": self._now_iso(),
                    "worker_started_at": None,
                    "worker_pid": None,
                }
            ),
            normalized_text=normalized,
            output_text_chunks=json.dumps(chunks),
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        return job

    def _update_job_progress(
        self,
        job: SynthesisJob,
        *,
        stage: str,
        percent: int,
        message: str,
        current_chunk: int = 0,
        total_chunks: int = 0,
        worker_pid: int | None = None,
        worker_started_at: str | None = None,
    ) -> None:
        request = json.loads(job.request_json or "{}")
        request["progress"] = {
            "stage": stage,
            "percent": max(0, min(100, int(percent))),
            "message": message,
            "current_chunk": current_chunk,
            "total_chunks": total_chunks,
        }
        request["last_heartbeat_at"] = self._now_iso()
        if worker_pid is not None:
            request["worker_pid"] = worker_pid
        if worker_started_at is not None:
            request["worker_started_at"] = worker_started_at
        job.request_json = json.dumps(request)
        self.db.add(job)
        self.db.commit()

    def run_job(self, job_id: str) -> SynthesisJob:
        job = self.db.get(SynthesisJob, job_id)
        if job is None:
            raise ValueError("Synthesis job not found")
        profile = self.db.get(VoiceProfile, job.voice_profile_id)
        if profile is None:
            job.status = "failed"
            self.db.add(job)
            self.db.commit()
            raise ValueError("Voice profile not found")

        request = json.loads(job.request_json or "{}")
        profile_report = self._profile_report(profile)
        reference_path, prompt_text = self._resolve_clone_reference(profile, profile_report)
        chunks = json.loads(job.output_text_chunks or "[]")
        if not chunks:
            job.status = "failed"
            self.db.add(job)
            self.db.commit()
            raise ValueError("Synthesis job has no text chunks")

        delivery_request = self.mastering.normalize_delivery_request(
            str(request.get("format", "wav")),
            int(request.get("sample_rate_hz", 24000)),
            int(request.get("channels", 1)),
        )
        engine_selection = self.engine_registry.select(
            job.mode,
            sample_rate_hz=int(delivery_request["sample_rate_hz"]),
            channels=int(delivery_request["channels"]),
        )
        engine = engine_selection["engine"]
        qc_plan = request.get("preflight_qc", {})
        regeneration_plan = request.get("regeneration_plan", {})
        normalized = job.normalized_text
        locale = str(request.get("locale", self.settings.xtts_default_language))
        require_native_master = bool(request.get("require_native_master", False))

        output_dir = Path("uploads") / profile.user_id / "generated" / job.id
        output_dir.mkdir(parents=True, exist_ok=True)
        chunk_dir = output_dir / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        native_mix_path = output_dir / "native-mix.wav"
        output_path = output_dir / f"output.{delivery_request['format']}"

        job.status = "running"
        self.db.add(job)
        self.db.commit()
        self._update_job_progress(job, stage="running", percent=5, message="Synthesis job started.", total_chunks=len(chunks))

        try:
            language_code = (locale or self.settings.xtts_default_language).split("-")[0].lower()
            render_chunks: list[str] = []
            for index, chunk in enumerate(chunks, start=1):
                render_text = chunk
                if index in qc_plan.get("failed_segments", []):
                    split_version = split_for_regeneration(chunk)
                    render_text = " ".join(split_version)
                render_chunks.append(render_text)

            isolated = self._run_isolated_engine_job(
                job=job,
                engine_name=engine.name,
                chunk_texts=render_chunks,
                chunk_dir=str(chunk_dir),
                speaker_wav=reference_path,
                language=language_code,
                prompt_text=prompt_text,
                voice_profile_report=profile_report,
            )
            chunk_paths = isolated["chunk_paths"]
            engine_runs = isolated["engine_runs"]

            self._update_job_progress(job, stage="mastering", percent=80, message="Concatenating and mastering generated chunks.")
            concat_info = self.mastering.concatenate_wav_chunks(chunk_paths, str(native_mix_path))
            delivery_report = self.mastering.master_audio(
                str(native_mix_path),
                str(output_path),
                audio_format=str(delivery_request["format"]),
                sample_rate_hz=int(delivery_request["sample_rate_hz"]),
                channels=int(delivery_request["channels"]),
            )

            if job.mode == "preview":
                asr_backcheck = {"status": "deferred", "reason": "Preview generation skips heavy ASR backcheck for responsiveness."}
                evaluation_payload = {"status": "deferred", "reason": "Preview generation skips heavy speaker/evaluation checks for responsiveness."}
            else:
                self._update_job_progress(job, stage="evaluation", percent=90, message="Running final ASR/evaluation checks.")
                asr_backcheck = self.asr_backcheck.evaluate(expected_text=normalized, chunks=chunks, audio_path=str(output_path))
                evaluation_report = self.evaluation.evaluate(
                    audio_path=str(output_path),
                    reference_path=reference_path,
                    expected_text=normalized,
                    chunks=chunks,
                )
                evaluation_payload = evaluation_report.to_dict()
        except SynthesisCancelledError as exc:
            job.status = "cancelled"
            self._update_job_progress(job, stage="cancelled", percent=100, message=str(exc))
            self.db.add(job)
            self.db.commit()
            return job
        except XTTSInferenceError as exc:
            job.status = "failed"
            self._update_job_progress(job, stage="failed", percent=100, message=str(exc))
            self.db.add(job)
            self.db.commit()
            return job
        except Exception as exc:
            job.status = "failed"
            self._update_job_progress(job, stage="failed", percent=100, message=f"Synthesis failed: {exc}")
            self.db.add(job)
            self.db.commit()
            return job

        asset = GeneratedAsset(
            synthesis_job_id=job.id,
            format=str(delivery_request["format"]),
            sample_rate=delivery_report["delivery"]["sample_rate_hz"],
            channels=delivery_report["delivery"]["channels"],
            duration_ms=delivery_report["delivery"]["duration_ms"],
            object_key=str(output_path),
            watermark_info_json=json.dumps(
                {
                    "provenance": "curated-clone-profile-v1",
                    "engine": engine_runs[0]["engine"] if engine_runs else engine.name,
                    "device": engine_runs[0].get("device") if engine_runs else None,
                    "language": engine_runs[0].get("language") if engine_runs else None,
                    "chunk_count": len(chunks),
                    "qc": qc_plan,
                    "regeneration_plan": regeneration_plan,
                    "asr_backcheck": asr_backcheck,
                    "evaluation": evaluation_payload,
                    "concat": concat_info,
                    "clone_profile": {
                        "dataset_status": profile_report.get("clone_dataset", {}).get("status"),
                        "curated_minutes": profile_report.get("clone_dataset", {}).get("curated_minutes"),
                        "reference_audio_path": reference_path,
                        "prompt_text_used": bool(prompt_text),
                    },
                    "engine_selection": {
                        "requested_mode": job.mode,
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
            self._update_job_progress(job, stage="failed", percent=100, message="Native master was required, but output is derived.")
            self.db.add(job)
            self.db.commit()
            return job

        self.db.add(asset)
        job.status = "completed"
        self._update_job_progress(job, stage="completed", percent=100, message="Synthesis completed. Download is ready.")
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
        try:
            self._mark_stale_if_needed(job)
        except Exception:
            # Preview/status should never 500 because stale checking failed.
            pass
        self.db.refresh(job)
        return {
            "job_id": job.id,
            "status": job.status,
            "preview_text": job.normalized_text,
            "chunks": json.loads(job.output_text_chunks),
            "request": json.loads(job.request_json),
        }

    def cancel_job(self, job_id: str) -> SynthesisJob:
        job = self.db.get(SynthesisJob, job_id)
        if job is None:
            raise ValueError("Synthesis job not found")
        request = json.loads(job.request_json or "{}")
        request["cancel_requested"] = True
        request["last_heartbeat_at"] = self._now_iso()
        request["progress"] = {
            "stage": "cancel_requested",
            "percent": int((request.get("progress") or {}).get("percent", 0) or 0),
            "message": "Cancellation requested. The synthesis worker will stop at the next safe checkpoint.",
            "current_chunk": int((request.get("progress") or {}).get("current_chunk", 0) or 0),
            "total_chunks": int((request.get("progress") or {}).get("total_chunks", 0) or 0),
        }
        job.request_json = json.dumps(request)
        if job.status in {"queued", "running"}:
            job.status = "cancel_requested"
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        return job

    def get_download_url(self, job_id: str) -> dict:
        asset = self.db.query(GeneratedAsset).filter(GeneratedAsset.synthesis_job_id == job_id).one_or_none()
        if asset is None:
            raise ValueError("Generated asset not found")
        metadata = json.loads(asset.watermark_info_json or "{}")
        return {
            "url": f"http://localhost:8000/v1/synthesis/{job_id}/file",
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
            "clone_profile": metadata.get("clone_profile", {}),
            "engine_selection": metadata.get("engine_selection", {}),
            "engine_registry": metadata.get("engine_registry", self.engine_registry.describe()),
        }

    def get_generated_file_path(self, job_id: str) -> str:
        asset = self.db.query(GeneratedAsset).filter(GeneratedAsset.synthesis_job_id == job_id).one_or_none()
        if asset is None:
            raise ValueError("Generated asset not found")
        path = Path(asset.object_key)
        if not path.exists() or not path.is_file():
            raise ValueError("Generated audio file not found on disk")
        return str(path)

    def _profile_report(self, profile: VoiceProfile) -> dict:
        try:
            return json.loads(profile.readiness_report_json or "{}")
        except json.JSONDecodeError:
            return {}

    def _resolve_clone_reference(self, profile: VoiceProfile, profile_report: dict) -> tuple[str, str]:
        prompt = profile_report.get("clone_dataset", {}).get("prompt", {})
        reference_path = (
            prompt.get("prompt_audio_path_16k")
            or prompt.get("prompt_audio_path")
            or prompt.get("single_prompt_audio_path")
            or profile.sample_audio_path
        )
        prompt_text = str(prompt.get("prompt_text") or "").strip()
        return str(reference_path or ""), prompt_text

    def _synthesis_progress_message(self, engine_name: str, index: int, total_chunks: int) -> str:
        if engine_name == "voxcpm2":
            return (
                f"Synthesizing chunk {index}/{total_chunks} with VoxCPM2. "
                "This is the primary clone model. First model load can take several minutes; CPU/Mac runs can be slow."
            )
        if engine_name == "chatterbox":
            return f"Synthesizing chunk {index}/{total_chunks} with Chatterbox CPU/GPU prompt cloning."
        return f"Synthesizing chunk {index}/{total_chunks}."

    def _run_isolated_engine_job(
        self,
        *,
        job: SynthesisJob,
        engine_name: str,
        chunk_texts: list[str],
        chunk_dir: str,
        speaker_wav: str,
        language: str,
        prompt_text: str,
        voice_profile_report: dict,
    ) -> dict:
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        payload = {
            "engine_name": engine_name,
            "voice_profile_id": job.voice_profile_id,
            "mode": job.mode,
            "chunk_texts": chunk_texts,
            "chunk_dir": chunk_dir,
            "speaker_wav": speaker_wav,
            "language": language,
            "prompt_text": prompt_text,
            "voice_profile_report": voice_profile_report,
        }
        process = ctx.Process(target=_synthesis_engine_worker, args=(payload, result_queue))
        process.start()
        worker_started_at = self._now_iso()
        self._update_job_progress(
            job,
            stage="synthesizing",
            percent=10,
            message=self._synthesis_progress_message(engine_name, 1, len(chunk_texts)),
            current_chunk=0,
            total_chunks=len(chunk_texts),
            worker_pid=process.pid or 0,
            worker_started_at=worker_started_at,
        )

        started_at = time.monotonic()
        heartbeat_interval = max(1, int(self.settings.synthesis_heartbeat_interval_seconds))
        timeout_seconds = max(int(self.settings.synthesis_chunk_timeout_seconds), len(chunk_texts) * heartbeat_interval * 3)

        try:
            while True:
                if self._cancel_requested(job.id):
                    self._terminate_process(process)
                    raise SynthesisCancelledError("Synthesis cancelled by user.")

                elapsed = int(time.monotonic() - started_at)
                if elapsed >= timeout_seconds:
                    self._terminate_process(process)
                    raise XTTSInferenceError(
                        f"Synthesis worker exceeded timeout of {timeout_seconds} seconds and was terminated. "
                        "The selected engine likely hung or crashed during model generation."
                    )

                try:
                    event = result_queue.get(timeout=heartbeat_interval)
                except Empty:
                    if not process.is_alive() and process.exitcode is not None:
                        break
                    self._update_job_progress(
                        job,
                        stage="synthesizing",
                        percent=min(75, 10 + int((elapsed / max(timeout_seconds, 1)) * 20)),
                        message=f"{self._synthesis_progress_message(engine_name, 1, len(chunk_texts))} Worker alive for {elapsed}s.",
                        current_chunk=int((json.loads(job.request_json or "{}").get("progress") or {}).get("current_chunk", 1) or 1),
                        total_chunks=len(chunk_texts),
                        worker_pid=process.pid or 0,
                    )
                    continue

                event_type = event.get("type")
                if event_type == "progress":
                    current_chunk = int(event.get("current_chunk") or 1)
                    total_chunks = int(event.get("total_chunks") or len(chunk_texts) or 1)
                    self._update_job_progress(
                        job,
                        stage="synthesizing",
                        percent=10 + int((current_chunk - 1) / max(total_chunks, 1) * 65),
                        message=self._synthesis_progress_message(engine_name, current_chunk, total_chunks),
                        current_chunk=current_chunk,
                        total_chunks=total_chunks,
                        worker_pid=process.pid or 0,
                    )
                    continue
                if event_type == "done":
                    process.join(timeout=1)
                    return {
                        "chunk_paths": list(event.get("chunk_paths") or []),
                        "engine_runs": list(event.get("engine_runs") or []),
                    }
                if event_type == "error":
                    self._terminate_process(process)
                    raise XTTSInferenceError(str(event.get("error") or "Isolated synthesis worker failed."))

            self._terminate_process(process)
            raise XTTSInferenceError(
                f"{engine_name} worker exited unexpectedly with code {process.exitcode}. "
                "This usually indicates a native runtime crash inside the model backend."
            )
        finally:
            try:
                result_queue.close()
            except Exception:
                pass

    def _terminate_process(self, process: mp.Process) -> None:
        if not process.is_alive():
            return
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            try:
                process.kill()
            except Exception:
                pass
            process.join(timeout=5)

    def _cancel_requested(self, job_id: str) -> bool:
        probe_db = SessionLocal()
        try:
            fresh = probe_db.get(SynthesisJob, job_id)
            if fresh is None:
                return False
            request = json.loads(fresh.request_json or "{}")
            return bool(request.get("cancel_requested"))
        finally:
            probe_db.close()

    def _mark_stale_if_needed(self, job: SynthesisJob) -> None:
        if job.status not in {"running", "cancel_requested"}:
            return
        request = json.loads(job.request_json or "{}")
        heartbeat_raw = request.get("last_heartbeat_at")
        heartbeat_at = self._parse_datetime(heartbeat_raw)
        if heartbeat_at is None:
            heartbeat_at = self._normalize_datetime(getattr(job, "updated_at", None))
        if heartbeat_at is None:
            return
        now = datetime.now(timezone.utc)
        if (now - heartbeat_at).total_seconds() < int(self.settings.synthesis_stale_seconds):
            return
        request["progress"] = {
            "stage": "failed",
            "percent": 100,
            "message": "Synthesis job became stale with no heartbeat updates. The engine worker likely crashed or was terminated.",
            "current_chunk": int((request.get("progress") or {}).get("current_chunk", 0) or 0),
            "total_chunks": int((request.get("progress") or {}).get("total_chunks", 0) or 0),
        }
        request["last_heartbeat_at"] = self._now_iso()
        job.request_json = json.dumps(request)
        job.status = "failed"
        self.db.add(job)
        self.db.commit()

    def _normalize_datetime(self, value) -> datetime | None:
        if value is None:
            return None
        if not isinstance(value, datetime):
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _parse_datetime(self, raw: str | None) -> datetime | None:
        if not raw:
            return None
        try:
            parsed = datetime.fromisoformat(str(raw))
        except Exception:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()
