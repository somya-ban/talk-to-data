/**
 * Typed Flask API client.
 *
 * Every function here corresponds to one endpoint in app.py.
 * No component should ever call fetch directly — always go through this layer.
 * That way, if Flask changes, only this file needs updating.
 */

import type {
  InitResponse,
  StatusResponse,
  ProposeMetricsResponse,
  ConfirmMetricsRequest,
  ConfirmMetricsResponse,
  GetMetricsResponse,
  QueryRequest,
  QueryResponse,
  FixRequest,
  FixResponse,
  HistoryResponse,
  ResetResponse,
  Metric,
} from "./types";

const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:5000";

/**
 * Generic typed fetch wrapper.
 * Throws on network errors. Returns parsed JSON on success.
 * Caller is responsible for inspecting `ok` field on the result.
 */
async function request<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_URL}${path}`;
  const headers: HeadersInit = {
    "Content-Type": "application/json",
    ...options.headers,
  };

  const response = await fetch(url, { ...options, headers });

  if (!response.ok && response.status >= 500) {
    // Server errors — bubble up as exceptions so error boundaries catch them.
    const text = await response.text();
    throw new Error(`Server error ${response.status}: ${text}`);
  }

  // For 4xx and 2xx, Flask returns a JSON envelope with `ok` field.
  // The caller decides what to do based on `ok`.
  return (await response.json()) as T;
}

// ─── Pipeline lifecycle ────────────────────────────────────────────────────────

export function initPipeline(apiKey?: string): Promise<InitResponse> {
  return request<InitResponse>("/api/init", {
    method: "POST",
    body: JSON.stringify(apiKey ? { groq_api_key: apiKey } : {}),
  });
}

export function getStatus(): Promise<StatusResponse> {
  return request<StatusResponse>("/api/status", { method: "GET" });
}

export function resetPipeline(): Promise<ResetResponse> {
  return request<ResetResponse>("/api/reset", { method: "POST" });
}

// ─── Metric flow ───────────────────────────────────────────────────────────────

export function proposeMetrics(): Promise<ProposeMetricsResponse> {
  return request<ProposeMetricsResponse>("/api/metrics/propose", {
    method: "POST",
  });
}

export function confirmMetrics(
  metrics: Metric[]
): Promise<ConfirmMetricsResponse> {
  const body: ConfirmMetricsRequest = { metrics };
  return request<ConfirmMetricsResponse>("/api/metrics/confirm", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function getMetrics(): Promise<GetMetricsResponse> {
  return request<GetMetricsResponse>("/api/metrics", { method: "GET" });
}

// ─── Query and correction ─────────────────────────────────────────────────────

export function runQuery(question: string): Promise<QueryResponse> {
  const body: QueryRequest = { question };
  return request<QueryResponse>("/api/query", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function fixQuery(
  question: string,
  correctedSql: string,
  originalSql?: string
): Promise<FixResponse> {
  const body: FixRequest = {
    question,
    corrected_sql: correctedSql,
    original_sql: originalSql,
  };
  return request<FixResponse>("/api/fix", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

// ─── History ──────────────────────────────────────────────────────────────────

export function getHistory(n: number = 10): Promise<HistoryResponse> {
  return request<HistoryResponse>(`/api/history?n=${n}`, { method: "GET" });
}

export function clearHistory(): Promise<{ ok: boolean; message: string }> {
  return request<{ ok: boolean; message: string }>("/api/history", {
    method: "DELETE",
  });
}

// ─── Streaming query ──────────────────────────────────────────────────────────

/**
 * Event types emitted by /api/query/stream.
 */
export type StreamEventType =
  | "stage_active"
  | "stage_complete"
  | "done"
  | "error";

export interface StreamEvent {
  event: StreamEventType;
  data: Record<string, unknown>;
}

/**
 * Run a query against the streaming endpoint.
 *
 * Uses fetch with a manual SSE reader because EventSource doesn't support POST.
 * This is the same pattern Vercel AI SDK uses for chat streaming.
 *
 * The onEvent callback fires for each event as it arrives. The promise
 * resolves when the stream ends (either via "done" or "error" event, or
 * when the connection closes).
 */
export async function runQueryStream(
  question: string,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const response = await fetch(`${API_URL}/api/query/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
    signal,
  });

  if (!response.ok) {
    throw new Error(`Stream request failed: ${response.status}`);
  }

  if (!response.body) {
    throw new Error("Response has no body");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // SSE events are separated by double newlines
      const events = buffer.split("\n\n");
      // Keep the last (potentially incomplete) chunk in the buffer
      buffer = events.pop() ?? "";

      for (const rawEvent of events) {
        if (!rawEvent.trim()) continue;
        const parsed = parseSSEEvent(rawEvent);
        if (parsed) {
          onEvent(parsed);
          // If the event is terminal, we can stop early
          if (parsed.event === "done" || parsed.event === "error") {
            return;
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * Parse a single SSE event block into structured form.
 * SSE format:
 *   event: stage_complete
 *   data: {"stage": "linking", ...}
 */
function parseSSEEvent(raw: string): StreamEvent | null {
  let eventType: string | null = null;
  let dataLine: string | null = null;

  for (const line of raw.split("\n")) {
    if (line.startsWith("event: ")) {
      eventType = line.slice(7).trim();
    } else if (line.startsWith("data: ")) {
      dataLine = line.slice(6).trim();
    }
  }

  if (!eventType || !dataLine) return null;

  try {
    const data = JSON.parse(dataLine) as Record<string, unknown>;
    return { event: eventType as StreamEventType, data };
  } catch {
    return null;
  }
}