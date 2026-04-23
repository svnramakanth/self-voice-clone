# self-voice-clone

Simple personal voice-clone MVP.

This project lets you:
- upload your audio sample,
- optionally provide matching transcript text or SRT/TXT,
- save a personal voice profile,
- prepare reference audio for future XTTS-style voice conditioning.

## Current behavior

The current app flow is simplified and usable.

That means:
- the app structure works,
- profile saving works,
- transcript can be omitted and will be auto-filled,
- uploaded audio is normalized, silence-trimmed, and loudness-normalized when possible,
- the app stores readiness/quality guidance based on reference duration,
- synthesis is now designed to use **XTTS v2** with processed reference audio,
- synthesis now renders chunk-by-chunk, measures the actual output file with `ffprobe`, and can export mastered **WAV** or **FLAC** delivery files,
- each delivery now includes a report telling you whether the file is a true native Spotify-style master or only a derived package from the XTTS native render,
- final-mode synthesis now fails closed by default when the selected engine cannot produce a true native master,
- enrollment now preserves the uploaded source recording separately from the XTTS conditioning derivative,
- mastering validation now includes measured integrated loudness and true-peak reporting when ffmpeg analysis is available,
- enrollment now stores real faster-whisper transcription when available, alignment estimates, and profile quality scoring,
- synthesis now records engine capabilities and a preflight long-form QC report before rendering,
- preview and final synthesis modes now resolve through an engine registry so the API can report engine capabilities explicitly,
- Phase B now includes distinct preview/final engine identities with per-request engine-selection warnings and capability reporting,
- B1-B4 now include engine selection rationale, fail-closed native-master enforcement, and a premium final-engine plugin slot,
- Phase C now adds expanded text normalization, pronunciation lexicon overrides, faster-whisper-backed ASR back-check when available, and selective regeneration planning,
- Phase D now adds release-grade mastering summaries, stronger delivery validation, and native-vs-derived truth reporting,
- Phase E now adds dependency-aware evaluation reports for similarity, intelligibility/WER, artifact scoring, human listening rubric, and golden-sample regression checks,
- but XTTS dependencies must be installed successfully in your API environment for real inference to work.

## Phase 1 conditioning rules

- low minimum allowed: around `15–30s`
- warn under about `2 minutes`
- recommend `5–10+ minutes` for better quality
- SRT is optional for basic inference flow
- if SRT/transcript is provided, it is stored for future alignment / dataset use
- if not provided, transcription fallback is used

## How to run

You need to start **2 servers**.

### 1) Start backend API

```bash
cd /Users/ramakanth/pers/self-voice-clone/vclone/apps/api
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn app.main:app --reload --port 8000
```

If XTTS installation is heavy on your machine, CPU inference may still work but can be slow. GPU can be enabled later via config.

Backend docs:
- http://localhost:8000/docs

### 2) Start frontend

Open a second terminal:

```bash
cd /Users/ramakanth/pers/self-voice-clone/vclone/apps/web
npm install
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/v1 npm run dev
```

Frontend:
- http://localhost:3000

## How to use

### Step 1: Create your voice profile

Open:
- http://localhost:3000/enrollment

Then:
- enter a profile name,
- upload one audio file,
- optionally paste transcript text or upload `.srt` / `.txt`,
- click **Save my voice profile**.

The enrollment response now includes a readiness report with:
- transcription metadata,
- alignment confidence,
- measured alignment metadata,
- estimated segment count,
- quality scoring,
- adaptation candidacy hint.

### Step 2: Check saved profiles

Open:
- http://localhost:3000/voices

You should see your saved profile listed there.

### Step 3: Generate output from text

Open:
- http://localhost:3000/synthesis

Then:
- choose your saved voice profile,
- type the text you want spoken,
- choose FLAC or WAV delivery settings,
- click **Generate**.

The synthesis response now includes:
- measured sample rate,
- measured channels,
- measured duration,
- checksum,
- Spotify-readiness report,
- release-grade delivery summary,
- ASR back-check,
- evaluation report,
- engine capability metadata,
- engine-selection warnings,
- engine-selection rationale,
- preview/final engine registry metadata,
- preflight chunk QC report.

`require_native_master=true` is now the default behavior for final delivery, so the API fails closed if the selected engine cannot satisfy the requested final delivery natively.

## UI guide

The web UI now contains an in-app documentation section on the home page explaining:
- how enrollment works,
- how backend analysis works,
- how synthesis/mastering works,
- and the current XTTS limitation for native Spotify-grade masters.

Recommended usage flow:
1. Start the backend server.
2. Start the frontend server.
3. Open `/enrollment` and upload a clean voice sample.
4. Review the returned readiness report.
5. Open `/voices` and confirm the saved profile exists.
6. Open `/synthesis`, choose the profile, enter text, and generate output.
7. Review the delivery report before publishing anything.

## Current simplified flow

The project intentionally does **not** require:
- auth,
- S3/MinIO,
- worker queues,
- production-grade liveness validation,
- true speaker verification,
- real Whisper/WhisperX ASR,
- forced alignment,
- full adaptation/fine-tuning,
- complex enrollment workflows.

## Where the main app lives

- app code: `/Users/ramakanth/pers/self-voice-clone/vclone`
- backend: `/Users/ramakanth/pers/self-voice-clone/vclone/apps/api`
- frontend: `/Users/ramakanth/pers/self-voice-clone/vclone/apps/web`

## XTTS Phase 2 note

Synthesis now targets XTTS v2 inference using:
- processed reference audio from the saved profile
- target text from the synthesis form

Important limitation:
- if XTTS natively renders mono / lower-sample-rate audio, exporting a 44.1/48 kHz stereo WAV/FLAC file is still only a **derived distribution master**, not a native high-resolution stereo capture. Final mode now rejects that output by default instead of treating it as release-ready.
- current transcription/back-check now attempts faster-whisper first, and speaker verification now attempts SpeechBrain first, but the repo still does not yet equal a full production WhisperX + premium final-render stack.
- Phase B is now implemented through B1-B4 in code: strict capability enforcement, stronger final-engine behavior, a configurable premium XTTS final engine profile, and richer UI/API reporting. However, even the premium XTTS path is still limited by XTTS’s native mono / lower-rate rendering ceiling.
- Phases C/D/E are implemented as a strong in-repo framework with dependency hooks for later replacement by real ASR, artifact models, and speaker-embedding evaluation. The current scores are useful for pipeline shaping, but they are not equal to external benchmark-grade evaluation yet.

If `TTS` / model initialization is unavailable, the API will now return a clear error instead of silently falling back to fake synthesis.
