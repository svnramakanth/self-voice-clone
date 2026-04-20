# vclone

MVP foundation for a self-voice-only voice cloning and TTS platform based on `docs/personal-voice-clone-tts-design.md`.

## What is implemented

- FastAPI backend with SQLite persistence
- Simple voice profile creation flow: upload audio + transcript text/SRT/TXT
- Local file storage for uploaded samples and generated outputs
- Saved voice profile listing
- Simple synthesis flow for entering text against a saved profile
- Mock synthesis engine placeholder so the UI/API flow works end to end

## Repo structure

- `apps/api` — FastAPI backend
- `apps/web` — Next.js frontend
- `docs/personal-voice-clone-tts-design.md` — original implementation design

## Backend run

```bash
cd /Users/ramakanth/pers/self-voice-clone/vclone/apps/api
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn app.main:app --reload --port 8000
```

API docs will be available at:

- `http://localhost:8000/docs`

## Frontend run

```bash
cd /Users/ramakanth/pers/self-voice-clone/vclone/apps/web
npm install
npm run dev
```

Frontend will be available at:

- `http://localhost:3000`

If your API runs elsewhere, set:

```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/v1
```

## Current MVP limitations

This is intentionally kept very simple.

The following are still mocked or simplified:

- no real authentication
- no S3/MinIO/object storage
- no worker queues
- no speaker/liveness validation
- no real voice-cloning inference yet
- generated output is still placeholder content until a real engine is integrated

## Simplified API routes

- `POST /v1/voice-profiles` — create a profile from uploaded audio + transcript text/file
- `GET /v1/voice-profiles`
- `GET /v1/voice-profiles/{voice_profile_id}`
- `POST /v1/synthesis`
- `GET /v1/synthesis/{job_id}/preview`
- `POST /v1/synthesis/{job_id}/download-url`

## Typical usage

1. Start backend API on port `8000`.
2. Start frontend on port `3000` with `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/v1`.
3. Open `http://localhost:3000/enrollment`.
4. Upload one audio sample.
5. Paste transcript text or upload an SRT/TXT file.
6. Save your voice profile.
7. Open `http://localhost:3000/voices` and confirm the profile exists.
8. Open `http://localhost:3000/synthesis`.
9. Select the saved profile, enter text, and generate output.

## Quick start

### Backend

```bash
cd /Users/ramakanth/pers/self-voice-clone/vclone/apps/api
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd /Users/ramakanth/pers/self-voice-clone/vclone/apps/web
npm install
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/v1 npm run dev
```

Then open:

- `http://localhost:3000`
- `http://localhost:8000/docs`

## If you want actual cloned voice output

The current app flow is now simple, but the engine is still mocked.
To get real audio in your own voice, the next step would be integrating an actual local voice cloning / TTS model behind the same simplified endpoints.