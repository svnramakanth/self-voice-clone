import { listVoiceProfiles } from "../../lib/api";
import { SectionCard } from "../../components/SectionCard";

export default async function VoicesPage() {
  const profiles = await listVoiceProfiles().catch(() => ({ items: [] }));

  return (
    <SectionCard title="Saved voice profiles">
      <p className="muted">Step 2: confirm your saved profile exists here. Then open the Generate page.</p>
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
              <td>{item.id}</td>
              <td><span className="pill">{item.status}</span></td>
              <td>{item.engine_family}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </SectionCard>
  );
}
