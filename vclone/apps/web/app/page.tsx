import { SectionCard } from "../components/SectionCard";
import Link from "next/link";

export default function HomePage() {
  return (
    <div className="grid">
      <section className="card hero stack">
        <div className="section-kicker">Simple personal voice workflow</div>
        <h1>Clean, minimal flow for saving your voice and generating speech</h1>
        <p className="muted">
          This UI is focused on three steps: upload your sample, confirm your saved profile,
          and generate output from text. The backend now attempts real XTTS synthesis, faster-whisper
          transcription/back-check, and speaker-verification hooks when the required ML dependencies are installed.
        </p>
        <div className="actions">
          <Link href="/enrollment" className="link-arrow">Start by saving your voice →</Link>
          <Link href="/synthesis" className="link-arrow">Go straight to generate →</Link>
        </div>
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-value">1</div>
            <div className="stat-label">Audio sample needed</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">3</div>
            <div className="stat-label">Simple steps in the flow</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">Local</div>
            <div className="stat-label">Storage for uploaded files</div>
          </div>
        </div>
      </section>

      <div className="grid cards">
        <SectionCard title="Save your voice profile" kicker="Step 1">
          <p className="muted">Upload your audio and provide matching transcript text or an SRT/TXT file.</p>
          <p><Link href="/enrollment" className="link-arrow">Open voice profile setup →</Link></p>
        </SectionCard>
        <SectionCard title="Check saved profiles" kicker="Step 2">
          <p className="muted">See the voice profiles you have already saved and confirm everything looks right.</p>
          <p><Link href="/voices" className="link-arrow">Open saved profiles →</Link></p>
        </SectionCard>
        <SectionCard title="Generate from text" kicker="Step 3">
          <p className="muted">Pick a saved profile, enter text, and generate output.</p>
          <p><Link href="/synthesis" className="link-arrow">Open text-to-speech →</Link></p>
        </SectionCard>
      </div>

      <SectionCard title="Before you use the app" kicker="Heads up">
        <div className="card-grid">
          <div className="info-tile">
            <h4>Run both servers first</h4>
            <div className="muted">Backend on <code>localhost:8000</code> and frontend on <code>localhost:3000</code>.</div>
          </div>
          <div className="info-tile">
            <h4>Current output is mocked</h4>
            <div className="muted">The flow can generate real XTTS-based audio if backend dependencies are installed, but true native stereo release masters still require a stronger final engine.</div>
          </div>
          <div className="info-tile">
            <h4>What this app optimizes for</h4>
            <div className="muted">Clarity, fewer steps, and a much simpler personal workflow.</div>
          </div>
        </div>
      </SectionCard>

      <SectionCard title="How the project works" kicker="Docs">
        <div className="feature-list">
          <div className="feature-item">
            <div className="feature-badge">1</div>
            <div>
              <strong>Enrollment</strong>
              <div className="muted">You upload one voice recording and optionally matching transcript text or SRT/TXT. The backend stores the original upload, creates a conditioning derivative, and generates readiness metadata.</div>
            </div>
          </div>
          <div className="feature-item">
            <div className="feature-badge">2</div>
            <div>
              <strong>Analysis</strong>
              <div className="muted">When available, the backend uses faster-whisper for transcription/back-check and attempts speaker-verification hooks. If those packages are missing, the app falls back to simpler metadata guidance.</div>
            </div>
          </div>
          <div className="feature-item">
            <div className="feature-badge">3</div>
            <div>
              <strong>Synthesis</strong>
              <div className="muted">Text is normalized, chunked, rendered chunk-by-chunk with XTTS, concatenated, mastered, and packaged as WAV/FLAC. Final mode fails closed if the result is not a true native master.</div>
            </div>
          </div>
          <div className="feature-item">
            <div className="feature-badge">4</div>
            <div>
              <strong>Current limitation</strong>
              <div className="muted">This repo still uses XTTS as the main synthesis engine, so perfect studio-grade native stereo Spotify delivery is not available without integrating a better final renderer.</div>
            </div>
          </div>
        </div>
      </SectionCard>
    </div>
  );
}
