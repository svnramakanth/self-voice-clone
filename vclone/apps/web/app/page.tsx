import { SectionCard } from "../components/SectionCard";
import Link from "next/link";

export default function HomePage() {
  return (
    <div className="grid cards">
      <SectionCard title="Step 1: Save your voice profile">
        <p className="muted">Upload your audio and provide matching transcript text or an SRT/TXT file.</p>
        <p><Link href="/enrollment">Open voice profile setup →</Link></p>
      </SectionCard>
      <SectionCard title="Step 2: Check saved profiles">
        <p className="muted">See the voice profiles you have already saved.</p>
        <p><Link href="/voices">Open saved profiles →</Link></p>
      </SectionCard>
      <SectionCard title="Step 3: Generate from text">
        <p className="muted">Pick a saved profile, enter text, and generate output.</p>
        <p><Link href="/synthesis">Open text-to-speech →</Link></p>
      </SectionCard>
      <SectionCard title="Before you open the UI">
        <p className="muted">Make sure backend is running on <code>localhost:8000</code> and frontend is running on <code>localhost:3000</code>.</p>
        <p className="muted">This version uses a mock engine, so it does not yet generate real cloned voice audio.</p>
      </SectionCard>
    </div>
  );
}
