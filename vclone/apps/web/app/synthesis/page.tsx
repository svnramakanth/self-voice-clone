"use client";

import { useEffect, useState } from "react";
import { cancelSynthesis, getSynthesisDownloadUrl, getSynthesisPreview, getSystemCapabilities, submitSynthesis, listVoiceProfiles } from "../../lib/api";
import { SectionCard } from "../../components/SectionCard";

export default function SynthesisPage() {
  const [job, setJob] = useState<any>(null);
  const [download, setDownload] = useState<any>(null);
  const [profiles, setProfiles] = useState<any[]>([]);
  const [capabilities, setCapabilities] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function waitForSynthesis(jobId: string) {
    for (;;) {
      const preview = (await getSynthesisPreview(jobId)) as any;
      setJob(preview);
      const status = String(preview.status || "").toLowerCase();
      if (status === "completed") {
        return preview;
      }
      if (status === "failed" || status === "cancelled") {
        const progress = preview.request?.progress;
        throw new Error(progress?.message || "Synthesis failed on the server.");
      }
      await new Promise((resolve) => window.setTimeout(resolve, 2500));
    }
  }

  useEffect(() => {
    listVoiceProfiles().then((response) => setProfiles(response.items ?? [])).catch(() => setProfiles([]));
    getSystemCapabilities().then(setCapabilities).catch(() => setCapabilities(null));
  }, []);

  return (
    <div className="grid two-col">
      <SectionCard title="Generate speech" kicker="Step 3">
        <div className="page-header">
          <h1>Turn text into output</h1>
          <p className="muted">Start with Preview mode for a reliable first voice test. The backend now prefers VoxCPM2 from your curated prompt bundle, with Chatterbox as fallback.</p>
        </div>
        <p className="muted">Final/master mode is stricter and can reject derived stereo exports. First confirm the actual voice identity in preview.</p>
        <p className="muted">If VoxCPM2/Chatterbox dependencies are missing, generation fails clearly instead of silently returning another bad XTTS clone.</p>
        <form
          onSubmit={async (event) => {
            event.preventDefault();
            setError(null);
            setJob(null);
            setDownload(null);
            setLoading(true);
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
              await waitForSynthesis(response.job_id);
              const downloadResponse = await getSynthesisDownloadUrl(response.job_id);
              setDownload(downloadResponse);
            } catch (submitError) {
              setError(submitError instanceof Error ? submitError.message : "Generation failed");
            } finally {
              setLoading(false);
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
            <select name="mode" defaultValue="preview">
              <option value="preview">Preview</option>
              <option value="final">Final master candidate</option>
            </select>
          </label>
          <label>
            Output format
            <select name="format" defaultValue="wav">
              <option value="wav">WAV</option>
              <option value="flac">FLAC</option>
            </select>
          </label>
          <label>
            Sample rate
            <select name="sample_rate_hz" defaultValue="24000">
              <option value="24000">24,000 Hz</option>
              <option value="44100">44,100 Hz</option>
              <option value="48000">48,000 Hz</option>
            </select>
          </label>
          <label>
            Channels
            <select name="channels" defaultValue="1">
              <option value="1">Mono</option>
              <option value="2">Stereo</option>
            </select>
          </label>
          <label>
            Locale
            <input name="locale" defaultValue="en-IN" />
          </label>
          <label>
            <input name="require_native_master" type="checkbox" />
            Require native master
          </label>
          <button type="submit" disabled={loading}>{loading ? "Generating..." : "Generate"}</button>
        </form>
        <div className="helper" style={{ marginTop: 16 }}>
          <strong>Delivery note</strong>
          <div className="muted">Recommended first test: Preview, WAV, mono, native master unchecked. Use final/master only after the voice actually sounds like you.</div>
          <div className="muted">If a job becomes stale or hangs, use Cancel and retry after restarting the API. Native model crashes now get detected more cleanly.</div>
        </div>
        {capabilities ? (
          <div className="result-box" style={{ marginTop: 16 }}>
            <strong>System capabilities</strong>
            <pre>{JSON.stringify(capabilities, null, 2)}</pre>
          </div>
        ) : null}
      </SectionCard>

      <SectionCard title="Generation output" kicker="Result">
        <div className="stack">
          <div className="helper">
            <strong>Clone pipeline</strong>
            <div className="muted">Primary: VoxCPM2 ultimate cloning from curated prompt audio + exact prompt text. Secondary: Chatterbox prompt cloning. XTTS is legacy only and disabled as silent fallback by default.</div>
            <div className="muted">On Mac, VoxCPM2 auto mode uses CUDA if present, otherwise CPU. MPS is opt-in only because this model can crash during MPSGraph warm-up.</div>
            {capabilities?.summary?.clone_pipeline ? <div className="muted">Runtime: {String(capabilities.summary.clone_pipeline)}</div> : null}
          </div>
          {error ? <div className="result-box"><pre>{error}</pre></div> : null}
          {job?.request?.progress ? (
            <div className="result-box">
              <strong>{job.request.progress.message}</strong>
              <div className="muted">Stage: {job.request.progress.stage}</div>
              <div className="muted">Progress: {job.request.progress.percent}%</div>
              <progress value={job.request.progress.percent} max={100} style={{ width: "100%", height: 14 }} />
              {job.request.progress.total_chunks ? (
                <div className="muted">Chunk: {job.request.progress.current_chunk}/{job.request.progress.total_chunks}</div>
              ) : null}
              {String(job.status || "").toLowerCase() === "running" || String(job.status || "").toLowerCase() === "cancel_requested" ? (
                <button
                  type="button"
                  style={{ marginTop: 12 }}
                  onClick={async () => {
                    try {
                      await cancelSynthesis(job.job_id);
                    } catch (cancelError) {
                      setError(cancelError instanceof Error ? cancelError.message : "Failed to cancel synthesis job.");
                    }
                  }}
                >
                  Cancel synthesis
                </button>
              ) : null}
            </div>
          ) : null}
          {download ? (
            <div className="result-box">
              <strong>Audio ready</strong>
              <div className="muted">{download.asset.format.toUpperCase()} • {download.asset.sample_rate_hz} Hz • {download.asset.channels} channel(s) • {(download.asset.duration_ms / 1000).toFixed(2)}s</div>
              <audio controls src={download.url} style={{ width: "100%", marginTop: 12 }} />
              <div className="actions" style={{ marginTop: 12 }}>
                <a className="link-arrow" href={download.url} download>Download audio</a>
                <a className="link-arrow" href={download.url} target="_blank" rel="noreferrer">Open audio</a>
              </div>
            </div>
          ) : null}
          {job ? (
            <details className="result-box">
              <summary>Job details</summary>
              <pre>{JSON.stringify(job, null, 2)}</pre>
            </details>
          ) : null}
          {download ? (
            <details className="result-box">
              <summary>Download and delivery details</summary>
              <pre>{JSON.stringify(download, null, 2)}</pre>
            </details>
          ) : null}
        </div>
      </SectionCard>
    </div>
  );
}