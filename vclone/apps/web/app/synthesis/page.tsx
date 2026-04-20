"use client";

import { useEffect, useState } from "react";
import { getSynthesisDownloadUrl, submitSynthesis, listVoiceProfiles } from "../../lib/api";
import { SectionCard } from "../../components/SectionCard";

export default function SynthesisPage() {
  const [job, setJob] = useState<any>(null);
  const [download, setDownload] = useState<any>(null);
  const [profiles, setProfiles] = useState<any[]>([]);

  useEffect(() => {
    listVoiceProfiles().then((response) => setProfiles(response.items ?? [])).catch(() => setProfiles([]));
  }, []);

  return (
    <div className="grid">
      <SectionCard title="Generate speech">
        <p className="muted">Step 3: select one saved voice profile, enter text, and generate output.</p>
        <form
          onSubmit={async (event) => {
            event.preventDefault();
            const data = new FormData(event.currentTarget);
            const response = await submitSynthesis({
              voice_profile_id: String(data.get("voice_profile_id")),
              text: String(data.get("text")),
              mode: "preview",
              format: String(data.get("format")),
              sample_rate_hz: Number(data.get("sample_rate_hz")),
              locale: String(data.get("locale")),
            });
            setJob(response);
            const downloadResponse = await getSynthesisDownloadUrl(response.job_id);
            setDownload(downloadResponse);
          }}
        >
          <select name="voice_profile_id" required defaultValue="">
            <option value="" disabled>Select your voice profile</option>
            {profiles.map((item) => (
              <option key={item.id} value={item.id}>{item.name} ({item.id.slice(0, 8)})</option>
            ))}
          </select>
          <textarea name="text" defaultValue="Hello from my personal voice model." rows={4} />
          <input name="format" defaultValue="wav" />
          <input name="sample_rate_hz" defaultValue="24000" />
          <input name="locale" defaultValue="en-IN" />
          <button type="submit">Generate</button>
        </form>
        {job ? <pre>{JSON.stringify(job, null, 2)}</pre> : null}
        {download ? <pre>{JSON.stringify(download, null, 2)}</pre> : null}
        <p className="muted">Note: current output is still a mock placeholder until a real voice engine is integrated.</p>
      </SectionCard>
    </div>
  );
}