from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

from app.core.config import get_settings


@dataclass
class DatasetRecord:
    index: int
    audio_path: str
    text: str
    duration_seconds: float
    word_count: int
    score: float
    source: str
    split: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VoiceDatasetBuilder:
    """Builds clone-ready profile artifacts from SRT-curated speech segments.

    This is intentionally separate from synthesis.  A high-quality clone starts
    with a clean, transcript-aligned segment dataset and a short exact
    prompt-audio/prompt-text pair.  The previous XTTS path mostly built one
    conditioning derivative; this builder creates the artifacts needed by
    VoxCPM2 ultimate cloning, Chatterbox prompt cloning, and future LoRA jobs.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.ffmpeg_path = shutil.which("ffmpeg")
        self.ffprobe_path = shutil.which("ffprobe")

    def build(
        self,
        *,
        source_audio_path: str,
        processed_audio_path: str,
        transcript_text: str,
        output_dir: str,
        curation_report: dict[str, Any] | None = None,
        progress_callback=None,
    ) -> dict[str, Any]:
        progress_callback = progress_callback or (lambda _update: None)
        dataset_dir = Path(output_dir) / "voice_dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = dataset_dir / "manifest.jsonl"
        train_path = dataset_dir / "train.jsonl"
        validation_path = dataset_dir / "validation.jsonl"
        test_path = dataset_dir / "test.jsonl"
        prompts_dir = dataset_dir / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)

        curation = curation_report or {}
        selected_segments = list(curation.get("selected_segments") or [])
        records, rejected = self._records_from_selected_segments(selected_segments)

        if not records:
            fallback_record = self._fallback_prompt_record(
                processed_audio_path=processed_audio_path,
                transcript_text=transcript_text,
                prompts_dir=prompts_dir,
            )
            if fallback_record:
                records = [fallback_record]
            else:
                rejected.append({"reason": "No usable SRT segments and fallback prompt extraction failed."})

        records = self._assign_splits(records)
        self._write_jsonl(manifest_path, [record.to_dict() for record in records])
        self._write_jsonl(train_path, [record.to_dict() for record in records if record.split == "train"])
        self._write_jsonl(validation_path, [record.to_dict() for record in records if record.split == "validation"])
        self._write_jsonl(test_path, [record.to_dict() for record in records if record.split == "test"])

        progress_callback(
            {
                "stage": "selecting_clone_prompt",
                "percent": 96,
                "message": "Selecting exact prompt audio/text for stable voice cloning.",
                "accepted_segments": len(records),
                "rejected_segments": len(rejected),
            }
        )
        prompt = self._build_prompt_artifacts(records=records, prompts_dir=prompts_dir)
        total_seconds = round(sum(record.duration_seconds for record in records), 2)
        train_seconds = round(sum(record.duration_seconds for record in records if record.split == "train"), 2)
        status = self._dataset_status(records, total_seconds, prompt)
        report = {
            "status": status,
            "purpose": "curated_voice_clone_dataset",
            "source_audio_path": source_audio_path,
            "processed_audio_path": processed_audio_path,
            "dataset_dir": str(dataset_dir),
            "manifest_path": str(manifest_path),
            "train_manifest_path": str(train_path),
            "validation_manifest_path": str(validation_path),
            "test_manifest_path": str(test_path),
            "accepted_segment_count": len(records),
            "rejected_segment_count": len(rejected),
            "curated_seconds": total_seconds,
            "train_seconds": train_seconds,
            "curated_minutes": round(total_seconds / 60.0, 2),
            "prompt": prompt,
            "engine_readiness": self._engine_readiness(records=records, total_seconds=total_seconds, prompt=prompt),
            "quality_policy": {
                "min_segment_seconds": self.settings.voice_dataset_min_segment_seconds,
                "max_segment_seconds": self.settings.voice_dataset_max_segment_seconds,
                "preferred_prompt_seconds": self.settings.voice_prompt_target_seconds,
                "ideal_training_minutes": "30-60 minutes curated single-speaker speech",
            },
            "rejected_preview": rejected[:50],
            "notes": [
                "This dataset is the source of truth for real cloning. Mastering cannot fix a bad speaker profile.",
                "VoxCPM2 ultimate cloning should use prompt.prompt_audio_path_16k plus prompt.prompt_text when available.",
                "Future LoRA/adaptation should train only from manifest/train/validation/test, not raw full-length audio.",
            ],
        }
        (dataset_dir / "curation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    def _records_from_selected_segments(self, selected_segments: list[dict[str, Any]]) -> tuple[list[DatasetRecord], list[dict[str, Any]]]:
        records: list[DatasetRecord] = []
        rejected: list[dict[str, Any]] = []
        for item in selected_segments:
            audio_path = item.get("segment_audio_path") or item.get("audio_path")
            text = self._clean_text(str(item.get("text") or ""))
            if not audio_path:
                rejected.append({"index": item.get("index"), "reason": "Missing extracted segment audio path."})
                continue
            path = Path(str(audio_path))
            if not path.exists() or path.stat().st_size <= 44:
                rejected.append({"index": item.get("index"), "reason": "Extracted segment audio is missing or empty.", "audio_path": str(path)})
                continue
            duration = self._duration_seconds(path) or round(float(item.get("duration_ms") or 0) / 1000.0, 3)
            word_count = len(text.split())
            reason = self._segment_rejection_reason(duration, word_count, text)
            if reason:
                rejected.append({"index": item.get("index"), "reason": reason, "audio_path": str(path), "duration_seconds": duration})
                continue
            score = self._segment_score(duration, word_count, item.get("speech_analysis") or {})
            records.append(
                DatasetRecord(
                    index=len(records) + 1,
                    audio_path=str(path),
                    text=text,
                    duration_seconds=round(duration, 3),
                    word_count=word_count,
                    score=score,
                    source="srt_curated_segment",
                    split="train",
                )
            )
        return records, rejected

    def _fallback_prompt_record(self, *, processed_audio_path: str, transcript_text: str, prompts_dir: Path) -> DatasetRecord | None:
        source = Path(processed_audio_path)
        if not source.exists():
            return None
        duration = min(max(self.settings.voice_prompt_target_seconds, 8), 30)
        output = prompts_dir / "fallback_prompt.wav"
        if self.ffmpeg_path:
            result = subprocess.run(
                [
                    self.ffmpeg_path,
                    "-y",
                    "-i",
                    str(source),
                    "-t",
                    str(duration),
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    str(output),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0 or not output.exists():
                return None
        else:
            shutil.copyfile(source, output)

        text = self._clean_text(" ".join((transcript_text or "").split()[:45]))
        if not text:
            text = "Reference speech prompt for personal voice cloning."
        actual_duration = self._duration_seconds(output) or float(duration)
        return DatasetRecord(
            index=1,
            audio_path=str(output),
            text=text,
            duration_seconds=round(actual_duration, 3),
            word_count=max(1, len(text.split())),
            score=0.55,
            source="fallback_processed_audio_prompt",
            split="train",
        )

    def _assign_splits(self, records: list[DatasetRecord]) -> list[DatasetRecord]:
        if len(records) < 10:
            return records
        for position, record in enumerate(records, start=1):
            if position % 20 == 0:
                record.split = "test"
            elif position % 10 == 0:
                record.split = "validation"
            else:
                record.split = "train"
        return records

    def _build_prompt_artifacts(self, *, records: list[DatasetRecord], prompts_dir: Path) -> dict[str, Any]:
        if not records:
            return {"status": "missing", "reason": "No accepted records available for prompt selection."}

        candidates = sorted(records, key=lambda record: (self._prompt_score(record), record.score), reverse=True)
        chosen: list[DatasetRecord] = []
        total = 0.0
        target = float(self.settings.voice_prompt_target_seconds)
        for record in candidates:
            if record.duration_seconds < 3.0 or record.duration_seconds > 15.0:
                continue
            chosen.append(record)
            total += record.duration_seconds
            if total >= target:
                break
        if not chosen:
            chosen = candidates[:1]
            total = sum(record.duration_seconds for record in chosen)

        first_prompt = prompts_dir / "best_prompt_01.wav"
        shutil.copyfile(chosen[0].audio_path, first_prompt)
        first_prompt_16k = prompts_dir / "best_prompt_01_16k.wav"
        first_text = prompts_dir / "best_prompt_01.txt"
        first_text.write_text(chosen[0].text, encoding="utf-8")

        prompt_pack = prompts_dir / "best_prompt_pack.wav"
        prompt_pack_16k = prompts_dir / "best_prompt_pack_16k.wav"
        single_prompt_text = chosen[0].text.strip()
        prompt_pack_text = " ".join(record.text for record in chosen).strip()

        if len(chosen) == 1:
            shutil.copyfile(chosen[0].audio_path, prompt_pack)
        else:
            self._concat_prompt_files([Path(record.audio_path) for record in chosen], prompt_pack)

        if self.ffmpeg_path:
            single_result = subprocess.run(
                [
                    self.ffmpeg_path,
                    "-y",
                    "-i",
                    str(first_prompt),
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    str(first_prompt_16k),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if single_result.returncode != 0 or not first_prompt_16k.exists():
                first_prompt_16k = first_prompt

            result = subprocess.run(
                [
                    self.ffmpeg_path,
                    "-y",
                    "-i",
                    str(prompt_pack),
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    str(prompt_pack_16k),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0 or not prompt_pack_16k.exists():
                prompt_pack_16k = prompt_pack
        else:
            first_prompt_16k = first_prompt
            prompt_pack_16k = prompt_pack

        (prompts_dir / "best_prompt_pack.txt").write_text(prompt_pack_text, encoding="utf-8")
        return {
            "status": "ready",
            "prompt_audio_path": str(first_prompt),
            "prompt_audio_path_16k": str(first_prompt_16k),
            "single_prompt_audio_path": str(first_prompt),
            "single_prompt_audio_path_16k": str(first_prompt_16k),
            "single_prompt_text_path": str(first_text),
            "prompt_text": single_prompt_text,
            "prompt_pack_audio_path": str(prompt_pack),
            "prompt_pack_audio_path_16k": str(prompt_pack_16k),
            "prompt_pack_text": prompt_pack_text,
            "prompt_segment_count": len(chosen),
            "prompt_seconds": round(chosen[0].duration_seconds, 2),
            "prompt_pack_seconds": round(total, 2),
            "prompt_records": [record.to_dict() for record in chosen],
        }

    def _concat_prompt_files(self, paths: list[Path], output_path: Path) -> None:
        if not self.ffmpeg_path:
            shutil.copyfile(paths[0], output_path)
            return
        concat_list = output_path.with_suffix(".concat.txt")
        concat_list.write_text("".join(f"file '{self._escape_concat_path(path)}'\n" for path in paths), encoding="utf-8")
        try:
            result = subprocess.run(
                [
                    self.ffmpeg_path,
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat_list),
                    "-ar",
                    "24000",
                    "-ac",
                    "1",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0 or not output_path.exists():
                shutil.copyfile(paths[0], output_path)
        finally:
            concat_list.unlink(missing_ok=True)

    def _dataset_status(self, records: list[DatasetRecord], total_seconds: float, prompt: dict[str, Any]) -> str:
        if prompt.get("status") != "ready":
            return "not_ready"
        if len(records) >= 250 and total_seconds >= 1800:
            return "adaptation_ready"
        if len(records) >= 40 and total_seconds >= 300:
            return "zero_shot_ready"
        return "limited_prompt_ready"

    def _engine_readiness(self, *, records: list[DatasetRecord], total_seconds: float, prompt: dict[str, Any]) -> dict[str, Any]:
        prompt_ready = prompt.get("status") == "ready" and bool(prompt.get("prompt_text"))
        return {
            "voxcpm2_ultimate_clone": "ready" if prompt_ready else "missing_prompt",
            "voxcpm2_lora_candidate": total_seconds >= 1800 and len(records) >= 250,
            "chatterbox_prompt_clone": "ready" if prompt_ready else "missing_prompt",
            "xtts_legacy_reference": "available" if prompt_ready else "fallback_only",
            "recommended_next_step": "run_voxcpm2_zero_shot_bakeoff" if prompt_ready else "record_or_curate_more_clean_speech",
        }

    def _segment_rejection_reason(self, duration_seconds: float, word_count: int, text: str) -> str | None:
        if duration_seconds < self.settings.voice_dataset_min_segment_seconds:
            return "Segment is too short for clone training/prompting."
        if duration_seconds > self.settings.voice_dataset_max_segment_seconds:
            return "Segment is too long for stable clone training/prompting."
        if word_count < 2:
            return "Transcript has too few words."
        if not text:
            return "Cleaned transcript is empty."
        return None

    def _segment_score(self, duration_seconds: float, word_count: int, speech_analysis: dict[str, Any]) -> float:
        duration_score = max(0.0, 1.0 - abs(duration_seconds - 8.0) / 12.0)
        word_score = max(0.0, min(1.0, word_count / 16.0))
        coverage = float(speech_analysis.get("speech_coverage_percent") or 80.0) / 100.0
        return round(max(0.05, min(0.99, (duration_score * 0.45) + (word_score * 0.25) + (coverage * 0.30))), 3)

    def _prompt_score(self, record: DatasetRecord) -> float:
        duration_preference = max(0.0, 1.0 - abs(record.duration_seconds - 9.0) / 10.0)
        word_preference = 1.0 if 6 <= record.word_count <= 35 else 0.55
        return (duration_preference * 0.55) + (word_preference * 0.25) + (record.score * 0.20)

    def _duration_seconds(self, path: Path) -> float:
        if not self.ffprobe_path:
            return 0.0
        result = subprocess.run(
            [
                self.ffprobe_path,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        try:
            return float((result.stdout or "").strip())
        except Exception:
            return 0.0

    def _write_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> None:
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    def _clean_text(self, text: str) -> str:
        cleaned = re.sub(r"<[^>]+>", " ", text or "")
        cleaned = re.sub(r"\{[^}]+\}", " ", cleaned)
        cleaned = cleaned.replace("&", " and ")
        cleaned = cleaned.replace("—", "-").replace("–", "-")
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        return cleaned.strip()

    def _escape_concat_path(self, path: Path) -> str:
        return path.resolve().as_posix().replace("'", "'\\''")