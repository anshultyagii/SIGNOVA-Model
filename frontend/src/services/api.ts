// SignSpeakAI Frontend - API Service
import type { HealthResponse, TranslateRequest, TranslateResponse } from "../types/api";

export const API_BASE_URL: string =
  (import.meta.env.VITE_API_BASE_URL as string) || "http://127.0.0.1:8000";

/**
 * Resolves the WebSocket URL for live sign recognition.
 * Ensures consistent connection to ws://127.0.0.1:8000/ws/recognize
 * unless overridden via environment variables.
 */
export function getWebSocketRecognizeUrl(): string {
  if (import.meta.env.VITE_WS_URL) {
    return import.meta.env.VITE_WS_URL as string;
  }
  if (import.meta.env.VITE_API_BASE_URL) {
    return `${(import.meta.env.VITE_API_BASE_URL as string).replace(/^http/, "ws")}/ws/recognize`;
  }
  return "ws://127.0.0.1:8000/ws/recognize";
}

export const WS_RECOGNIZE_URL: string = getWebSocketRecognizeUrl();

export class ApiError extends Error {
  statusCode?: number;
  detail?: string;

  constructor(message: string, statusCode?: number, detail?: string) {
    super(message);
    this.name = "ApiError";
    this.statusCode = statusCode;
    this.detail = detail;
  }
}

/**
 * Checks connectivity and retrieves application status from backend GET /health.
 */
export async function checkHealth(): Promise<HealthResponse> {
  const url = `${API_BASE_URL}/health`;
  try {
    const response = await fetch(url, {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
    });

    if (!response.ok) {
      throw new ApiError(
        `Health check failed with status ${response.status}`,
        response.status
      );
    }

    return (await response.json()) as HealthResponse;
  } catch (err: unknown) {
    if (err instanceof ApiError) {
      throw err;
    }
    throw new ApiError(
      `Cannot connect to SignSpeakAI backend at ${API_BASE_URL}. Ensure FastAPI is running.`,
      0
    );
  }
}

/**
 * Translates an ordered list of glosses using backend POST /translate.
 */
export async function translateGlosses(
  glosses: string[]
): Promise<TranslateResponse> {
  const url = `${API_BASE_URL}/translate`;
  const payload: TranslateRequest = { glosses };

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      let detail = `Error ${response.status}`;
      try {
        const errorJson = await response.json();
        if (typeof errorJson?.detail === "string") {
          detail = errorJson.detail;
        } else if (Array.isArray(errorJson?.detail)) {
          detail = errorJson.detail.map((d: any) => d.msg || JSON.stringify(d)).join("; ");
        }
      } catch {
        // Fallback to status text
        detail = response.statusText || detail;
      }
      throw new ApiError(detail, response.status, detail);
    }

    return (await response.json()) as TranslateResponse;
  } catch (err: unknown) {
    if (err instanceof ApiError) {
      throw err;
    }
    throw new ApiError(
      `Network error contacting ${API_BASE_URL}/translate. Is the backend running?`,
      0
    );
  }
}
