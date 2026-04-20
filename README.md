# self-voice-clone

Simple personal voice-clone MVP.

This project lets you:
- upload your audio sample,
- provide matching transcript text or SRT/TXT,
- save a personal voice profile,
- enter text later and generate output.

## Important limitation

The current app flow is simplified and usable, but the voice engine is still a **mock placeholder**.

That means:
- the app structure works,
- profile saving works,
- text submission works,
- but it does **not yet generate real cloned audio in your voice**.

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
- either paste transcript text or upload `.srt` / `.txt`,
- click **Save my voice profile**.

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
- click **Generate**.

## Current simplified flow

The project intentionally does **not** require:
- auth,
- S3/MinIO,
- worker queues,
- liveness validation,
- speaker verification,
- complex enrollment workflows.

## Where the main app lives

- app code: `/Users/ramakanth/pers/self-voice-clone/vclone`
- backend: `/Users/ramakanth/pers/self-voice-clone/vclone/apps/api`
- frontend: `/Users/ramakanth/pers/self-voice-clone/vclone/apps/web`

## Next real step if you want actual cloning

To get real audio in your voice, the next step is integrating a real local/open-source voice cloning or TTS engine behind the same simplified upload + synthesize flow.
