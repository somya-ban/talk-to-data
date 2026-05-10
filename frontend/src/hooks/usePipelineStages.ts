/**
 * Fake client-side pipeline stage cycling.
 *
 * Backend /api/query is a single synchronous round-trip — we never get
 * incremental updates. To make the wait feel intentional and educational,
 * cycle through 4 stages on a 600-1200ms timer while the real query
 * is in flight. When the real response arrives, populate detail strings
 * from response data and snap to all-complete.
 *
 * This is Option A — purely cosmetic stage faking. Locked decision.
 */

import { useState, useCallback, useRef, useEffect } from "react";
import type { PipelineStage } from "@/components/chat/PipelineStatus";

const STAGE_LABELS = [
  "Finding the right tables",
  "Composing the query",
  "Running the query",
  "Summarising the results",
] as const;

const STAGE_DURATIONS_MS = [800, 700, 1100, 600];

interface UsePipelineStagesResult {
  /** Stages array to pass to <PipelineStatus />. null when idle. */
  stages: PipelineStage[] | null;
  /** Begin the staged animation. Call when query is fired. */
  start: () => void;
  /** Complete all stages with optional detail strings. Call when response arrives. */
  complete: (details?: {
    tables?: string[];
    metricName?: string | null;
    rowCount?: number;
  }) => void;
  /** Clear state. Call when AssistantTurn is rendered (PendingTurn resolved). */
  reset: () => void;
}

export function usePipelineStages(): UsePipelineStagesResult {
  const [stages, setStages] = useState<PipelineStage[] | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const indexRef = useRef(0);

  // Clear any active timer on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  const advance = useCallback(() => {
    setStages((prev) => {
      if (prev === null) return null;
      const next = prev.map((s, i): PipelineStage => {
        if (i < indexRef.current) return { ...s, status: "complete" };
        if (i === indexRef.current) return { ...s, status: "active" };
        return { ...s, status: "pending" };
      });
      return next;
    });

    if (indexRef.current < STAGE_LABELS.length - 1) {
      const dur = STAGE_DURATIONS_MS[indexRef.current];
      timerRef.current = setTimeout(() => {
        indexRef.current += 1;
        advance();
      }, dur);
    }
    // When we reach the last stage we hold there until complete() is called.
  }, []);

  const start = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    indexRef.current = 0;
    setStages(
      STAGE_LABELS.map(
        (label, i): PipelineStage => ({
          label,
          status: i === 0 ? "active" : "pending",
        }),
      ),
    );
    const dur = STAGE_DURATIONS_MS[0];
    timerRef.current = setTimeout(() => {
      indexRef.current = 1;
      advance();
    }, dur);
  }, [advance]);

  const complete = useCallback(
    (details?: {
      tables?: string[];
      metricName?: string | null;
      rowCount?: number;
    }) => {
      if (timerRef.current) clearTimeout(timerRef.current);

      const tablesDetail =
        details?.tables && details.tables.length > 0
          ? details.tables.slice(0, 3)
          : undefined;
      const metricDetail = details?.metricName
        ? `Using your definition of ${details.metricName}`
        : undefined;
      const rowDetail =
        typeof details?.rowCount === "number"
          ? `${details.rowCount} ${details.rowCount === 1 ? "row" : "rows"}`
          : undefined;

      setStages([
        { label: STAGE_LABELS[0], detail: tablesDetail, status: "complete" },
        { label: STAGE_LABELS[1], detail: metricDetail, status: "complete" },
        { label: STAGE_LABELS[2], detail: rowDetail, status: "complete" },
        { label: STAGE_LABELS[3], status: "complete" },
      ]);
    },
    [],
  );

  const reset = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    indexRef.current = 0;
    setStages(null);
  }, []);

  return { stages, start, complete, reset };
}