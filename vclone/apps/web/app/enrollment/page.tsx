"use client";

import { useState } from "react";
import { createSimpleVoiceProfile } from "../../lib/api";
import { SectionCard } from "../../components/SectionCard";

export default function EnrollmentPage() {
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  return (
    <div className="grid two-col">
      <SectionCard title="Create your voice profile" kicker="Step 1">
        <div className="page-header">
          <h1>Save your voice sample</h1>
          <p className="muted">Give the app one voice sample for conditioning. Transcript text is optional — if you skip it, the app will generate fallback transcript metadata, estimate alignment, and score the profile for stability.</p>
        </div>
        <p className="muted">
          Upload one clear voice sample. We will normalize it, trim silence, apply loudness normalization, estimate alignment quality, and score whether the profile is strong enough for higher-quality synthesis workflows.
        </p>
        <form
          onSubmit={async (event) => {
            event.preventDefault();
            setLoading(true);
            const data = new FormData(event.currentTarget);
            const response = await createSimpleVoiceProfile({
              name: String(data.get("name") || "My Voice"),
              transcript_text: String(data.get("transcript_text") || ""),
              audio_file: data.get("audio_file") as File,
              transcript_file: (data.get("transcript_file") as File | null) || null,
            });
            setResult(response);
            setLoading(false);
          }}
        >
          <label>
            Profile name
            <input name="name" defaultValue="My Voice" placeholder="Profile name" />
          </label>
          <label>
            Audio sample
            <input name="audio_file" type="file" accept="audio/*" required />
          </label>
          <label>
            Transcript text
            <textarea name="transcript_text" rows={7} placeholder="Optional: paste transcript text here, or upload an SRT/TXT file below." />
          </label>
          <label>
            Transcript file (optional)
            <input name="transcript_file" type="file" accept=".srt,.txt,text/plain,application/x-subrip" />
          </label>
          <button type="submit" disabled={loading}>{loading ? "Saving..." : "Save my voice profile"}</button>
        </form>
        {result ? <div className="result-box"><pre>{JSON.stringify(result, null, 2)}</pre></div> : null}
      </SectionCard>

      <SectionCard title="Tips for better input" kicker="Guidance">
        <div className="feature-list">
          <div className="feature-item">
            <div className="feature-badge">1</div>
            <div>
              <strong>Use clean audio.</strong>
              <div className="muted">15–30 seconds is allowed, under 2 minutes gets a warning, and 5–10+ minutes is recommended for better conditioning quality.</div>
            </div>
          </div>
          <div className="feature-item">
            <div className="feature-badge">2</div>
            <div>
              <strong>Transcript should match exactly.</strong>
              <div className="muted">If you provide text or SRT, it should correspond closely to what is spoken in the audio so the alignment and quality scores stay high.</div>
            </div>
          </div>
          <div className="feature-item">
            <div className="feature-badge">3</div>
            <div>
              <strong>What next?</strong>
              <div className="muted">After saving, go to <code>Saved Profiles</code>, then open <code>Generate</code>.</div>
            </div>
          </div>
        </div>
      </SectionCard>
    </div>
  );
}
