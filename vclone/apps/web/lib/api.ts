const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/v1";

type ApiError = { detail?: string; message?: string };

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

export type UploadSessionResponse = {
  upload_id: string;
  filename: string;
  size_bytes: number;
  chunk_size: number;
  total_chunks: number;
  received_chunks: number[];
  received_bytes: number;
  status: string;
  stage?: string | null;
  srt_offset_ms: number;
  processing_percent: number;
  processing_message?: string | null;
  processing_attempt: number;
  accepted_segments: number;
  rejected_segments: number;
  current_segment_index: number;
  total_segments: number;
  last_updated_at?: string | null;
  error?: string | null;
  voice_profile_id?: string | null;
};

export type UploadProgress = {
  loaded: number;
  total: number;
  percent: number;
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

export type SystemCapabilitiesResponse = {
  status: string;
  engines: Record<string, unknown>;
  summary: Record<string, unknown>;
};

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const error = (await response.json().catch(() => null)) as ApiError | null;
    throw new Error(error?.detail ?? error?.message ?? `${response.status} ${response.statusText || "Request failed"}`);
  }
  return response.json() as Promise<T>;
}

async function apiFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(input, init);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Network request failed";
    throw new Error(
      `${message}. Could not reach the API at ${API_BASE}. Make sure the FastAPI server is running and CORS is enabled for the web app origin.`
    );
  }
}

export async function createSimpleVoiceProfile(payload: {
  name: string;
  transcript_text: string;
  audio_file: File;
  transcript_file?: File | null;
  onUploadProgress?: (progress: UploadProgress) => void;
}): Promise<SimpleVoiceProfileResponse> {
  if (!(payload.audio_file instanceof File) || payload.audio_file.size === 0) {
    throw new Error("Please choose a valid audio file before submitting.");
  }

  const formData = new FormData();
  formData.append("name", payload.name);
  formData.append("transcript_text", payload.transcript_text);
  formData.append("audio_file", payload.audio_file);
  if (payload.transcript_file) {
    formData.append("transcript_file", payload.transcript_file);
  }

  return new Promise<SimpleVoiceProfileResponse>((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", `${API_BASE}/voice-profiles`);
    request.responseType = "json";

    request.upload.onprogress = (event) => {
      if (!payload.onUploadProgress || !event.lengthComputable) {
        return;
      }
      payload.onUploadProgress({
        loaded: event.loaded,
        total: event.total,
        percent: Math.min(100, Math.round((event.loaded / event.total) * 100)),
      });
    };

    request.onerror = () => {
      reject(new Error("Network error while uploading audio. Please check whether the API is reachable and try again."));
    };

    request.onload = () => {
      const status = request.status;
      const responseData = request.response ?? null;

      if (status >= 200 && status < 300) {
        payload.onUploadProgress?.({ loaded: 1, total: 1, percent: 100 });
        resolve(responseData as SimpleVoiceProfileResponse);
        return;
      }

      const error = responseData as ApiError | null;
      reject(new Error(error?.detail ?? error?.message ?? `${status} ${request.statusText || "Request failed"}`));
    };

    request.send(formData);
  });
}

export async function createUploadSession(payload: {
  name: string;
  filename: string;
  content_type: string;
  size_bytes: number;
  transcript_text: string;
  srt_offset_ms: number;
}): Promise<UploadSessionResponse> {
  const response = await apiFetch(`${API_BASE}/uploads/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return readJson<UploadSessionResponse>(response);
}

export async function getUploadSession(uploadId: string): Promise<UploadSessionResponse> {
  const response = await apiFetch(`${API_BASE}/uploads/sessions/${uploadId}`, { cache: "no-store" });
  return readJson<UploadSessionResponse>(response);
}

export async function uploadSessionChunk(
  uploadId: string,
  chunkIndex: number,
  chunk: Blob,
  onProgress?: (progress: UploadProgress) => void
): Promise<UploadSessionResponse> {
  return new Promise<UploadSessionResponse>((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("PUT", `${API_BASE}/uploads/sessions/${uploadId}/chunks/${chunkIndex}`);
    request.responseType = "json";
    request.setRequestHeader("Content-Type", "application/octet-stream");

    request.upload.onprogress = (event) => {
      if (!onProgress || !event.lengthComputable) return;
      onProgress({
        loaded: event.loaded,
        total: event.total,
        percent: Math.min(100, Math.round((event.loaded / event.total) * 100)),
      });
    };

    request.onerror = () => reject(new Error(`Network error while uploading chunk ${chunkIndex + 1}.`));
    request.ontimeout = () => reject(new Error(`Timed out while uploading chunk ${chunkIndex + 1}.`));
    request.onload = () => {
      const responseData = request.response ?? null;
      if (request.status >= 200 && request.status < 300) {
        onProgress?.({ loaded: chunk.size, total: chunk.size, percent: 100 });
        resolve(responseData as UploadSessionResponse);
        return;
      }

      const error = responseData as ApiError | null;
      reject(new Error(error?.detail ?? error?.message ?? `${request.status} ${request.statusText || "Chunk upload failed"}`));
    };

    request.send(chunk);
  });
}

export async function uploadSessionTranscript(uploadId: string, file: File): Promise<UploadSessionResponse> {
  const response = await apiFetch(`${API_BASE}/uploads/sessions/${uploadId}/transcript/${encodeURIComponent(file.name)}`, {
    method: "PUT",
    headers: { "Content-Type": file.type || "text/plain" },
    body: file,
  });
  return readJson<UploadSessionResponse>(response);
}

export async function completeUploadSession(uploadId: string): Promise<{ upload_id: string; status: string; message: string }> {
  const response = await apiFetch(`${API_BASE}/uploads/sessions/${uploadId}/complete`, { method: "POST" });
  return readJson(response);
}

export async function retryUploadProcessing(uploadId: string): Promise<{ upload_id: string; status: string; message: string }> {
  const response = await apiFetch(`${API_BASE}/uploads/sessions/${uploadId}/retry-processing`, { method: "POST" });
  return readJson(response);
}

export async function cancelUploadProcessing(uploadId: string): Promise<{ upload_id: string; status: string; message: string }> {
  const response = await apiFetch(`${API_BASE}/uploads/sessions/${uploadId}/cancel`, { method: "POST" });
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
  const response = await apiFetch(`${API_BASE}/synthesis`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return readJson<SynthesisJobResponse>(response);
}

export async function getSynthesisPreview(jobId: string) {
  const response = await apiFetch(`${API_BASE}/synthesis/${jobId}/preview`, { cache: "no-store" });
  return readJson(response);
}

export async function getSynthesisDownloadUrl(jobId: string): Promise<SynthesisDownloadResponse> {
  const response = await apiFetch(`${API_BASE}/synthesis/${jobId}/download-url`, {
    method: "POST",
  });
  const payload = await readJson<SynthesisDownloadResponse>(response);
  if (!payload.url || payload.url.includes("storage.local")) {
    payload.url = `${API_BASE}/synthesis/${jobId}/file`;
  }
  return payload;
}

export async function getSystemCapabilities(): Promise<SystemCapabilitiesResponse> {
  const response = await fetch(`${API_BASE}/system/capabilities`, { cache: "no-store" });
  return readJson<SystemCapabilitiesResponse>(response);
}
