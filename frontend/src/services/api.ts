import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

export interface HealthResponse {
  status: string;
  device: string;
  model_loaded: boolean;
}

export interface VoiceInfo {
  name: string;
  count: number;
  files: string[];
}

export interface VoicesResponse {
  voices: VoiceInfo[];
}

export const healthCheck = async (): Promise<HealthResponse> => {
  const response = await api.get<HealthResponse>("/api/health");
  return response.data;
};

export const generateVoice = async (
  text: string,
  referenceAudio: File | null,
  voiceName: string
): Promise<Blob> => {
  const formData = new FormData();
  formData.append("text", text);
  formData.append("voice_name", voiceName);

  if (referenceAudio) {
    formData.append("reference_audio", referenceAudio);
  }

  const response = await axios.post(`${API_BASE_URL}/api/generate`, formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
    responseType: "blob",
  });

  return response.data;
};

export const listVoices = async (): Promise<VoicesResponse> => {
  const response = await api.get<VoicesResponse>("/api/voices");
  return response.data;
};

export default api;
