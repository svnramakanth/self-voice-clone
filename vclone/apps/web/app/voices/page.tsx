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
              </tr>
            </thead>
            <tbody>
              {(profiles.items ?? []).map((item: any) => (
                <tr key={item.id}>
                  <td>{item.name}</td>
                  <td><code>{item.id}</code></td>
                  <td><span className="pill">{item.status}</span></td>
                  <td>{item.engine_family}</td>
                </tr>
              ))}
              {!(profiles.items ?? []).length ? (
                <tr>
                  <td colSpan={4} className="muted">No profiles yet. Create one on the enrollment page first.</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="What this page tells you" kicker="Quick note">
        <div className="helper">
          <strong>If the profile is visible, your upload step worked.</strong>
          <div className="muted">You can now go to Generate and pick this profile from the dropdown. The status and engine column come directly from the backend voice profile response.</div>
        </div>
      </SectionCard>
    </div>
  );
}
