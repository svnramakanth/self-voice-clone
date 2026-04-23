"use client";

import { useEffect, useState } from "react";
import { getSynthesisDownloadUrl, submitSynthesis, listVoiceProfiles } from "../../lib/api";
import { SectionCard } from "../../components/SectionCard";

export default function SynthesisPage() {
  const [job, setJob] = useState<any>(null);
  const [download, setDownload] = useState<any>(null);
  const [profiles, setProfiles] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listVoiceProfiles().then((response) => setProfiles(response.items ?? [])).catch(() => setProfiles([]));
  }, []);

  return (
    <div className="grid two-col">
      <SectionCard title="Generate speech" kicker="Step 3">
        <div className="page-header">
          <h1>Turn text into output</h1>
          <p className="muted">Generate a mastered WAV/FLAC delivery file. Final mode now fails closed unless the output validates as a true native master rather than a derived XTTS package.</p>
        </div>
        <p className="muted">Choose a saved profile, enter your text, and render a final output. If XTTS only produces mono 24 kHz internally, final mode will now reject the request instead of pretending that a packaged export is studio-native.</p>
        <p className="muted">The request now also includes a preflight long-form QC pass so risky chunks can be identified before final packaging.</p>
        <p className="muted">Preview and final modes now flow through an engine registry with distinct preview/final engine identities, warnings, and capability reporting.</p>
        <p className="muted">Native-master enforcement is enabled by default. Preview mode remains useful for experimentation, but final release output requires a truly native-capable engine.</p>
        <p className="muted">Generation also reports measured mastering data plus clearly labeled heuristic ASR/evaluation placeholders so you can distinguish real validation from scaffolding.</p>
        <form
          onSubmit={async (event) => {
            event.preventDefault();
            setError(null);
            setJob(null);
            setDownload(null);
            try {
              const data = new FormData(event.currentTarget);
              const response = await submitSynthesis({
                voice_profile_id: String(data.get("voice_profile_id")),
                text: String(data.get("text")),
                mode: String(data.get("mode")),
                format: String(data.get("format")),
                sample_rate_hz: Number(data.get("sample_rate_hz")),
                channels: Number(data.get("channels")),
                locale: String(data.get("locale")),
                require_native_master: data.get("require_native_master") !== null,
              });
              setJob(response);
              const downloadResponse = await getSynthesisDownloadUrl(response.job_id);
              setDownload(downloadResponse);
            } catch (submitError) {
              setError(submitError instanceof Error ? submitError.message : "Generation failed");
            }
          }}
        >
          <label>
            Voice profile
            <select name="voice_profile_id" required defaultValue="">
              <option value="" disabled>Select your voice profile</option>
              {profiles.map((item) => (
                <option key={item.id} value={item.id}>{item.name} ({item.id.slice(0, 8)})</option>
              ))}
            </select>
          </label>
          <label>
            Text to generate
            <textarea name="text" defaultValue="Hello from my personal voice model." rows={6} />
          </label>
          <label>
            Render mode
            <select name="mode" defaultValue="final">
              <option value="preview">Preview</option>
              <option value="final">Final master candidate</option>
            </select>
          </label>
          <label>
            Output format
            <select name="format" defaultValue="flac">
              <option value="flac">FLAC</option>
              <option value="wav">WAV</option>
            </select>
          </label>
          <label>
            Sample rate
            <select name="sample_rate_hz" defaultValue="48000">
              <option value="24000">24,000 Hz</option>
              <option value="44100">44,100 Hz</option>
              <option value="48000">48,000 Hz</option>
            </select>
          </label>
          <label>
            Channels
            <select name="channels" defaultValue="2">
              <option value="1">Mono</option>
              <option value="2">Stereo</option>
            </select>
          </label>
          <label>
            Locale
            <input name="locale" defaultValue="en-IN" />
          </label>
          <label>
            <input name="require_native_master" type="checkbox" defaultChecked />
            Require native master
          </label>
          <button type="submit">Generate</button>
        </form>
        <div className="helper" style={{ marginTop: 16 }}>
          <strong>Delivery note</strong>
          <div className="muted">A 48 kHz stereo FLAC/WAV export is not accepted as a studio master here unless the engine rendered it natively. Derived upsampled or dual-mono exports are rejected in final mode.</div>
        </div>
      </SectionCard>

      <SectionCard title="Generation output" kicker="Result">
        <div className="stack">
          <div className="helper">
            <strong>Current limitation</strong>
            <div className="muted">Synthesis still depends on XTTS backend availability, and XTTS remains natively mono 24 kHz in this repo. So preview is usable, but true Spotify-grade final release requires a better final engine.</div>
          </div>
          {error ? <div className="result-box"><pre>{error}</pre></div> : null}
          {job ? <div className="result-box"><pre>{JSON.stringify(job, null, 2)}</pre></div> : null}
          {download ? <div className="result-box"><pre>{JSON.stringify(download, null, 2)}</pre></div> : null}
        </div>
      </SectionCard>
    </div>
  );
}