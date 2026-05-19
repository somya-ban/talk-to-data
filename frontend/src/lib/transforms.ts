/**
 * Shape transforms between backend JSON and frontend component props.
 *
 * Backend types in types.ts mirror Flask's response envelope exactly.
 * Frontend components have their own narrower or differently-shaped props.
 * This file is the single place where translation happens — never inline
 * a transform inside a component.
 */

import type { QueryResponse } from "./types";
import type { ResponseData } from "@/components/result/ResponseTurn";
import type { MaskedColumn } from "@/components/result/PrivacyPanel";

/**
 * Backend returns masked_columns as ["Customer.name", "Customer.email"].
 * PrivacyPanel wants [{column, pattern}]. Pattern is derived from the
 * column name's last segment after the dot — that's what guard.py
 * matches against (the bare column name like "name" or "email").
 */
function toMaskedColumns(columns: string[]): MaskedColumn[] {
  return columns.map((col) => {
    const lastSegment = col.includes(".") ? col.split(".").pop()! : col;
    return { column: col, pattern: lastSegment };
  });
}

/**
 * Translate a Flask /api/query response into the ResponseData shape
 * that ResponseTurn consumes.
 */
export function queryResponseToResponseData(r: QueryResponse): ResponseData {
  return {
    narration: r.answer.narration,
    metric: r.metric_used
      ? {
          name: r.metric_used.name,
          description: r.metric_used.description,
          sql_formula: r.metric_used.sql_formula,
        }
      : null,
    chart:
      r.answer.chart_data.length > 0
        ? {
            chart_type: r.answer.chart_type,
            data: r.answer.chart_data,
            columns: r.answer.columns,
            x_key: r.answer.x_key ?? undefined,
            y_key: r.answer.y_key ?? undefined,
          }
        : null,
    sql: r.sql,
    corrections: r.corrections.map((c) => ({
      attempt: c.attempt,
      sql: c.sql,
      error_type: c.error_type,
      error_message: c.error_message,
    })),
    masked_columns: toMaskedColumns(r.privacy.masked_columns),
  };
}