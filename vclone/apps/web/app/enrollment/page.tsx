"use client";

import { useState } from "react";
import { createSimpleVoiceProfile } from "../../lib/api";
import { SectionCard } from "../../components/SectionCard";

export default function EnrollmentPage() {
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  return (
    <div className="grid">
      <SectionCard title="Create my voice profile">
        <p className="muted">
          Step 1: upload one audio file and provide the matching transcript text or an SRT/TXT file.
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
          <input name="name" defaultValue="My Voice" placeholder="Profile name" />
          <input name="audio_file" type="file" accept="audio/*" required />
          <textarea name="transcript_text" rows={6} placeholder="Paste transcript text here, or upload an SRT/TXT file below." />
          <input name="transcript_file" type="file" accept=".srt,.txt,text/plain,application/x-subrip" />
          <button type="submit" disabled={loading}>{loading ? "Saving..." : "Save my voice profile"}</button>
        </form>
        {result ? <pre>{JSON.stringify(result, null, 2)}</pre> : null}
        <p className="muted">After saving, go to the Saved Profiles page, then open Generate.</p>
      </SectionCard>
    </div>
  );
}
