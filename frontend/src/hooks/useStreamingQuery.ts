/**
 * Streaming query hook — replaces usePipelineStages for the real backend
 * SSE-based pipeline events.
 *
 * Each stage transitions through pending → active → complete based on
 * actual events emitted by the Flask backend. The detail strings are
 * populated from real backend data, not faked.
 *
 * Dwell-time enforcement:
 *   Backend stages can complete in <100ms (the LLM call itself, executing
 *   pre-validated SQL on DuckDB). Human reading speed is much slower. We
 *   enforce a minimum dwell time per stage so the user can read each
 *   resolved detail before the next stage advances. Real data, humane pace.
 */

import { useState, useCallback, useRef, useEffect } from "react";
import type { PipelineStage } from "@/components/chat/PipelineStatus";
import type { QueryResponse } from "@/lib/types";
import { runQueryStream, type StreamEvent } from "@/lib/api";

const STAGE_ORDER = ["linking", "generating", "executing", "narrating"] as const;

type StageId = (typeof STAGE_ORDER)[number];

const STAGE_LABELS: Record<StageId, string> = {
  linking: "Finding the right tables",
  generating: "Composing the query",
  executing: "Running the query",
  narrating: "Summarising the results",
};

/**
 * Minimum time each stage must remain visible in any state (active or complete)
 * before another event can advance it. Tuned per-stage based on how much detail
 * each one surfaces — stages with richer detail get longer dwell time.
 */
const MIN_STAGE_DWELL_MS = 700;

interface UseStreamingQueryResult {
  stages: PipelineStage[] | null;
  run: (question: string) => Promise<QueryResponse | null>;
  reset: () => void;
}

export function useStreamingQuery(): UseStreamingQueryResult {
  const [stages, setStages] = useState<PipelineStage[] | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setStages(null);
  }, []);

  const run = useCallback(
    async (question: string): Promise<QueryResponse | null> => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setStages(
        STAGE_ORDER.map(
          (id): PipelineStage => ({
            label: STAGE_LABELS[id],
            status: "pending",
          })
        )
      );

      let finalResponse: QueryResponse | null = null;

      // Queue of stage events buffered from backend. The drain loop pops
      // one at a time and applies it with a minimum dwell between applies.
      // This decouples backend event pace from frontend display pace.
      const eventQueue: StreamEvent[] = [];
      let queueDrainResolved = false;
      let lastApplyTime = 0;
      let drainPromise: Promise<void> | null = null;

      const applyEvent = (event: StreamEvent) => {
        if (event.event === "stage_active") {
          const stage = event.data.stage as StageId;
          setStages((prev) => {
            if (!prev) return prev;
            return prev.map((s, idx): PipelineStage => {
              const stageId = STAGE_ORDER[idx];
              if (stageId === stage) {
                return { ...s, status: "active" };
              }
              return s;
            });
          });
        } else if (event.event === "stage_complete") {
          const stage = event.data.stage as StageId;
          const detail = formatStageDetail(stage, event.data);
          setStages((prev) => {
            if (!prev) return prev;
            return prev.map((s, idx): PipelineStage => {
              const stageId = STAGE_ORDER[idx];
              if (stageId === stage) {
                return { ...s, status: "complete", detail };
              }
              return s;
            });
          });
        }
      };

      // Drain the queue at humane pace — one event every MIN_STAGE_DWELL_MS,
      // unless the backend is naturally slower (then we wait for the next event).
      const drainQueue = async () => {
        while (!queueDrainResolved || eventQueue.length > 0) {
          if (eventQueue.length === 0) {
            // Wait briefly for more events
            await new Promise((r) => setTimeout(r, 50));
            continue;
          }

          const event = eventQueue.shift()!;
          const now = Date.now();
          const elapsed = now - lastApplyTime;

          // Only enforce dwell time on stage_complete events (those have detail
          // to read). stage_active events flow at backend pace.
          if (event.event === "stage_complete" && elapsed < MIN_STAGE_DWELL_MS) {
            await new Promise((r) =>
              setTimeout(r, MIN_STAGE_DWELL_MS - elapsed)
            );
          }

          applyEvent(event);
          lastApplyTime = Date.now();
        }
      };

      drainPromise = drainQueue();

      const handleEvent = (event: StreamEvent) => {
        if (event.event === "stage_active" || event.event === "stage_complete") {
          eventQueue.push(event);
        } else if (event.event === "done") {
          finalResponse = event.data as unknown as QueryResponse;
        } else if (event.event === "error") {
          finalResponse = {
            ok: false,
            success: false,
            error: (event.data.error as string) ?? "Streaming error",
            answer: {
              narration: "",
              chart_type: "table",
              chart_data: [],
              columns: [],
              x_key: null,
              y_key: null,
              row_count: 0,
            },
            metric_used: null,
            sql: "",
            was_corrected: false,
            corrections: [],
            privacy: { masked_columns: [] },
            timestamp: new Date().toISOString(),
          } as QueryResponse;
        }
      };

      try {
        await runQueryStream(question, handleEvent, controller.signal);
      } catch (err) {
        if ((err as Error).name === "AbortError") {
          queueDrainResolved = true;
          await drainPromise;
          return null;
        }
        queueDrainResolved = true;
        await drainPromise;
        throw err;
      }

      // Backend is done. Signal queue drain to finish, then wait for it.
      queueDrainResolved = true;
      await drainPromise;

      return finalResponse;
    },
    []
  );

  return { stages, run, reset };
}

// ─── Stage detail formatters ──────────────────────────────────────────────────

function formatStageDetail(
  stage: StageId,
  data: Record<string, unknown>
): string | string[] | undefined {
  if (stage === "linking") {
    const tables = data.tables as string[] | undefined;
    if (tables && tables.length > 0) {
      const shown = tables.slice(0, 3);
      if (tables.length > 3) {
        return [...shown, `+${tables.length - 3}`];
      }
      return shown;
    }
    return undefined;
  }

  if (stage === "generating") {
    const attempts = data.attempts as number | undefined;
    if (typeof attempts === "number" && attempts > 1) {
      return `${attempts} attempts`;
    }
    return undefined;
  }

  if (stage === "executing") {
    const rowCount = data.row_count as number | undefined;
    const wasCorrected = data.was_corrected as boolean | undefined;
    const correctionCount = data.correction_count as number | undefined;

    if (wasCorrected && correctionCount && correctionCount > 0) {
      const rowStr =
        typeof rowCount === "number"
          ? `${rowCount} ${rowCount === 1 ? "row" : "rows"}`
          : "";
      return rowStr
        ? `${rowStr} · ${correctionCount} correction${correctionCount > 1 ? "s" : ""}`
        : `${correctionCount} correction${correctionCount > 1 ? "s" : ""}`;
    }
    if (typeof rowCount === "number") {
      return `${rowCount} ${rowCount === 1 ? "row" : "rows"}`;
    }
    return undefined;
  }

  if (stage === "narrating") {
    const metricUsed = data.metric_used as string | null | undefined;
    if (metricUsed) {
      return `Using ${metricUsed}`;
    }
    return undefined;
  }

  return undefined;
}