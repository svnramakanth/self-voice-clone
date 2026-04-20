const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/v1";

type ApiError = { detail?: string };

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const error = (await response.json().catch(() => ({}))) as ApiError;
    throw new Error(error.detail ?? "Request failed");
  }
  return response.json() as Promise<T>;
}

export async function createSimpleVoiceProfile(payload: {
  name: string;
  transcript_text: string;
  audio_file: File;
  transcript_file?: File | null;
}) {
  const formData = new FormData();
  formData.append("name", payload.name);
  formData.append("transcript_text", payload.transcript_text);
  formData.append("audio_file", payload.audio_file);
  if (payload.transcript_file) {
    formData.append("transcript_file", payload.transcript_file);
  }

  const response = await fetch(`${API_BASE}/voice-profiles`, {
    method: "POST",
    body: formData,
  });
  return readJson(response);
}

export type EnrollmentPayload = {
  locale: string;
  consent_text_version: string;
  intended_use: string;
};

export async function createEnrollment(payload: EnrollmentPayload) {
  const response = await fetch(`${API_BASE}/enrollments`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return response.json();
}

export async function presignAudio(enrollmentId: string, payload: { filename: string; content_type: string; size_bytes: number }) {
  const response = await fetch(`${API_BASE}/enrollments/${enrollmentId}/audio:presign`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return response.json();
}

export async function validateEnrollment(enrollmentId: string, payload: { audio_asset_ids: string[]; transcript_asset_ids: string[] }) {
  const response = await fetch(`${API_BASE}/enrollments/${enrollmentId}/validate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return response.json();
}

export async function createVoiceProfile(enrollmentId: string, payload: { mode: string; engine_preference: string; allow_adaptation: boolean }) {
  const response = await fetch(`${API_BASE}/voice-profiles/create/from-enrollment/${enrollmentId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return response.json();
}

export async function listVoiceProfiles() {
  const response = await fetch(`${API_BASE}/voice-profiles`, { cache: "no-store" });
  return readJson(response);
}

export async function submitSynthesis(payload: {
  voice_profile_id: string;
  text: string;
  mode: string;
  format: string;
  sample_rate_hz: number;
  locale: string;
}) {
  const response = await fetch(`${API_BASE}/synthesis`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return readJson(response);
}

export async function getSynthesisPreview(jobId: string) {
  const response = await fetch(`${API_BASE}/synthesis/${jobId}/preview`, { cache: "no-store" });
  return readJson(response);
}

export async function getSynthesisDownloadUrl(jobId: string) {
  const response = await fetch(`${API_BASE}/synthesis/${jobId}/download-url`, {
    method: "POST",
  });
  return readJson(response);
}
