/**
 * TypeScript interfaces mirroring the JSON contract from app.py.
 *
 * These types are the source of truth for every API response shape.
 * If Flask changes a response structure, update the type here first,
 * then let TypeScript guide you to every component that needs adjustment.
 */

// ─── Domain types ──────────────────────────────────────────────────────────────

export type ChartType = "line" | "bar" | "table" | "stat";

export type ErrorTypeLabel =
  | "SYNTAX_ERROR"
  | "COLUMN_NOT_FOUND"
  | "TABLE_NOT_FOUND"
  | "EMPTY_RESULT"
  | "TYPE_ERROR"
  | "AMBIGUOUS_COLUMN"
  | "JOIN_ERROR"
  | "UNKNOWN_ERROR";

export interface Metric {
  name: string;
  description: string;
  sql_formula: string;
  tables: string[];
  columns: string[];
}

export interface MetricProposal extends Metric {}

export interface CorrectionEntry {
  attempt: number;
  sql: string;
  error_type: string;
  error_message: string;
}

export interface ChartDataRow {
  [columnName: string]: string | number | boolean | null;
}

// ─── /api/init ─────────────────────────────────────────────────────────────────

export interface InitResponse {
  ok: boolean;
  message?: string;
  error?: string;
  detail?: string;
  schema?: {
    tables: number;
    relations: number;
    services: string[];
  };
  graph?: {
    nodes: number;
    edges: number;
  };
  embeddings?: {
    ddl: number;
    documentation: number;
    qa_pairs: number;
  };
  duckdb?: {
    tables_seeded: number;
    row_counts: Record<string, number>;
  };
  metrics_loaded?: number;
}

// ─── /api/status ───────────────────────────────────────────────────────────────

export interface StatusResponse {
  ok: boolean;
  ready: boolean;
  metrics_confirmed?: number;
  qa_pairs?: number;
  metrics_yaml_exists?: boolean;
}

// ─── /api/metrics/propose ──────────────────────────────────────────────────────

export interface ProposeMetricsResponse {
  ok: boolean;
  proposals?: MetricProposal[];
  count?: number;
  error?: string;
}

// ─── /api/metrics/confirm ──────────────────────────────────────────────────────

export interface ConfirmMetricsRequest {
  metrics: Metric[];
}

export interface ConfirmMetricsResponse {
  ok: boolean;
  message?: string;
  count?: number;
  error?: string;
}

// ─── /api/metrics ──────────────────────────────────────────────────────────────

export interface GetMetricsResponse {
  ok: boolean;
  metrics: Metric[];
  count: number;
}

// ─── /api/query ────────────────────────────────────────────────────────────────

export interface QueryRequest {
  question: string;
}

export interface QueryAnswer {
  narration: string;
  chart_type: ChartType;
  chart_data: ChartDataRow[];
  columns: string[];
  x_key: string | null;
  y_key: string | null;
  row_count: number;
}

export interface QueryResponse {
  ok: boolean;
  success: boolean;
  answer: QueryAnswer;
  metric_used: Metric | null;
  sql: string;
  was_corrected: boolean;
  corrections: CorrectionEntry[];
  privacy: {
    masked_columns: string[];
  };
  timestamp: string;
  error?: string;
  detail?: string;
}

// ─── /api/fix ──────────────────────────────────────────────────────────────────

export interface FixRequest {
  question: string;
  corrected_sql: string;
  original_sql?: string;
}

export interface FixResponse {
  ok: boolean;
  message?: string;
  rows?: number;
  error?: string;
  sql?: string;
}

// ─── /api/history ──────────────────────────────────────────────────────────────

export interface HistoryEntry {
  question: string;
  sql: string;
  narration: string;
  chart_type: ChartType;
  row_count: number;
  was_corrected: boolean;
  timestamp: string;
  success: boolean;
}

export interface HistoryResponse {
  ok: boolean;
  history: HistoryEntry[];
  total: number;
}

// ─── /api/reset ────────────────────────────────────────────────────────────────

export interface ResetResponse {
  ok: boolean;
  message: string;
}

// ─── Conversation turn types (frontend-only, not from Flask) ───────────────────

export type TurnRole = "user" | "assistant";

export interface UserTurn {
  id: string;
  role: "user";
  question: string;
  timestamp: string;
}

export interface AssistantTurn {
  id: string;
  role: "assistant";
  response: QueryResponse;
}

export interface PendingTurn {
  id: string;
  role: "assistant";
  pending: true;
  stage: "linking" | "generating" | "executing" | "narrating";
}

export type Turn = UserTurn | AssistantTurn | PendingTurn;