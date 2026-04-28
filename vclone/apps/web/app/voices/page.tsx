import { listVoiceProfiles } from "../../lib/api";
import { SectionCard } from "../../components/SectionCard";

export default async function VoicesPage() {
  const profiles = await listVoiceProfiles().catch(() => ({ items: [] }));

  return (
    <div className="grid">
      <SectionCard title="Saved voice profiles" kicker="Step 2">
        <div className="page-header">
          <h1>Your saved voices</h1>
          <p className="muted">Every uploaded profile appears here before you generate text from it.</p>
        </div>
        <p className="muted">Confirm your saved profile exists here, then move to the Generate page.</p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>ID</th>
                <th>Status</th>
                <th>Engine</th>
                <th>Clone dataset</th>
                <th>Prompt</th>
              </tr>
            </thead>
            <tbody>
              {(profiles.items ?? []).map((item: any) => {
                const cloneDataset = item.readiness_report?.clone_dataset ?? {};
                const prompt = cloneDataset.prompt ?? {};
                return (
                  <tr key={item.id}>
                    <td>{item.name}</td>
                    <td><code>{item.id}</code></td>
                    <td><span className="pill">{item.status}</span></td>
                    <td>{item.engine_family}</td>
                    <td>{cloneDataset.status ?? "not built"} {cloneDataset.curated_minutes ? `• ${cloneDataset.curated_minutes} min` : ""}</td>
                    <td>{prompt.status ?? "missing"} {prompt.prompt_seconds ? `• ${prompt.prompt_seconds}s` : ""}</td>
                  </tr>
                );
              })}
              {!(profiles.items ?? []).length ? (
                <tr>
                  <td colSpan={6} className="muted">No profiles yet. Create one on the enrollment page first.</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="What this page tells you" kicker="Quick note">
        <div className="helper">
          <strong>If the profile is visible, your upload step worked.</strong>
          <div className="muted">For a real clone, also check that clone dataset and prompt are ready. VoxCPM2 uses the prompt audio plus exact prompt text for stable voice identity.</div>
        </div>
      </SectionCard>
    </div>
  );
}
