import { SectionCard } from "../components/SectionCard";
import Link from "next/link";

export default function HomePage() {
  return (
    <div className="grid">
      <section className="card hero stack">
        <div className="section-kicker">Simple personal voice workflow</div>
        <h1>Clean, minimal flow for saving your voice and generating speech</h1>
        <p className="muted">
          This UI is now focused on just three steps: upload your sample, confirm your saved profile,
          and generate output from text. No extra enterprise workflow noise.
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
            <div className="muted">The flow works, but this version does not yet generate real cloned voice audio.</div>
          </div>
          <div className="info-tile">
            <h4>What this app optimizes for</h4>
            <div className="muted">Clarity, fewer steps, and a much simpler personal workflow.</div>
          </div>
        </div>
      </SectionCard>
    </div>
  );
}
