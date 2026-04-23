const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/v1";

type ApiError = { detail?: string };

export type VoiceProfileSummary = {
  id: string;
  name: string;
  enrollment_id: string;
  status: string;
  engine_family: string;
  base_model_version: string;
  readiness_report: Record<string, unknown>;
};

export type VoiceProfileListResponse = {
  items: VoiceProfileSummary[];
};

export type SimpleVoiceProfileResponse = {
  voice_profile_id: string;
  status: string;
  name: string;
  readiness_report?: Record<string, unknown> | null;
};

export type SynthesisJobResponse = {
  job_id: string;
  status: string;
  message?: string | null;
};

export type SynthesizedAssetInfo = {
  format: string;
  sample_rate_hz: number;
  channels: number;
  duration_ms: number;
  local_path: string;
  checksum_sha256: string;
};

export type SynthesisDownloadResponse = {
  url: string;
  expires_in_seconds: number;
  asset: SynthesizedAssetInfo;
  delivery_report: Record<string, unknown>;
  evaluation: Record<string, unknown>;
  asr_backcheck: Record<string, unknown>;
  engine_selection: Record<string, unknown>;
  engine_registry: Record<string, unknown>;
};

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
}): Promise<SimpleVoiceProfileResponse> {
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
  return readJson<SimpleVoiceProfileResponse>(response);
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

export async function listVoiceProfiles(): Promise<VoiceProfileListResponse> {
  const response = await fetch(`${API_BASE}/voice-profiles`, { cache: "no-store" });
  return readJson<VoiceProfileListResponse>(response);
}

export async function submitSynthesis(payload: {
  voice_profile_id: string;
  text: string;
  mode: string;
  format: string;
  sample_rate_hz: number;
  channels: number;
  locale: string;
  require_native_master?: boolean;
}): Promise<SynthesisJobResponse> {
  const response = await fetch(`${API_BASE}/synthesis`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return readJson<SynthesisJobResponse>(response);
}

export async function getSynthesisPreview(jobId: string) {
  const response = await fetch(`${API_BASE}/synthesis/${jobId}/preview`, { cache: "no-store" });
  return readJson(response);
}

export async function getSynthesisDownloadUrl(jobId: string): Promise<SynthesisDownloadResponse> {
  const response = await fetch(`${API_BASE}/synthesis/${jobId}/download-url`, {
    method: "POST",
  });
  return readJson<SynthesisDownloadResponse>(response);
}
