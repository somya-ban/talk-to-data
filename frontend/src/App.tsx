import { useEffect, useState } from "react";
import { AppShell } from "@/components/shell/AppShell";
import { ConversationView } from "@/components/chat/ConversationView";
import type { ConversationTurn } from "@/components/chat/ConversationView";
import { MetricScreen } from "@/components/metrics/MetricScreen";
import type { MetricDefinition } from "@/components/metrics/MetricCard";

import {
  initPipeline,
  getStatus,
  proposeMetrics,
  confirmMetrics,
  getMetrics,
} from "@/lib/api";
import { queryResponseToResponseData } from "@/lib/transforms";
import { useStreamingQuery } from "@/hooks/useStreamingQuery";
import { useMetricsStore } from "@/store/metricsStore";
import { useStatusStore } from "@/store/statusStore";

import { Loader2 } from "lucide-react";

// ─── Top-level screen state ──────────────────────────────────────────────────

type Screen = "boot" | "boot_error" | "metric_setup" | "chat" | "metric_manage";

// ─── Component ───────────────────────────────────────────────────────────────

function App() {
  const [screen, setScreen] = useState<Screen>("boot");
  const [bootError, setBootError] = useState<string | null>(null);

  // Conversation state — local to App; we don't need a Zustand store here
  // because turns are owned by the visible chat session and don't cross trees.
  const [turns, setTurns] = useState<ConversationTurn[]>([]);
  const [proposedMetrics, setProposedMetrics] = useState<MetricDefinition[]>([]);

  const setMetricsConfirmed = useStatusStore((s) => s.setMetricsConfirmed);
  const setMetrics = useMetricsStore((s) => s.setMetrics);

  const pipeline = useStreamingQuery();

  // ── Boot sequence ──────────────────────────────────────────────────────────

  useEffect(() => {
    let cancelled = false;

    async function boot() {
      try {
        // Step 1: init pipeline. The Flask app reads GROQ_API_KEY from .env,
        // so we don't pass one from the frontend.
        const initRes = await initPipeline();
        if (cancelled) return;
        if (!initRes.ok) {
          setBootError(initRes.error ?? "Pipeline failed to initialise.");
          setScreen("boot_error");
          return;
        }

        // Step 2: check status — does this user already have confirmed metrics?
        const statusRes = await getStatus();
        if (cancelled) return;
        const confirmed = statusRes.metrics_confirmed ?? 0;
        setMetricsConfirmed(confirmed);

        if (confirmed === 0) {
          // Step 3: first run — propose metrics, show MetricScreen.
          const proposalRes = await proposeMetrics();
          if (cancelled) return;
          if (!proposalRes.ok || !proposalRes.proposals) {
            // Proposing failed — skip metric setup and go straight to chat.
            // The user can still ask questions; they just won't have inline
            // metric attribution in responses. Acceptable degraded mode.
            setScreen("chat");
            return;
          }
          setProposedMetrics(
            proposalRes.proposals.map((p) => ({
              name: p.name,
              description: p.description,
              sql_formula: p.sql_formula,
            })),
          );
          setScreen("metric_setup");
        } else {
          // Returning user — straight to chat.
          setScreen("chat");
        }
      } catch (err) {
        if (cancelled) return;
        const msg = err instanceof Error ? err.message : "Unknown boot error";
        setBootError(msg);
        setScreen("boot_error");
      }
    }

    boot();
    return () => {
      cancelled = true;
    };
  }, [setMetricsConfirmed]);

  // ── Metric confirmation flow ──────────────────────────────────────────────

  async function handleConfirmMetrics(confirmed: MetricDefinition[]) {
    // Translate frontend MetricDefinition into the Metric shape Flask expects.
    // Empty tables/columns are fine — sql_formula is the source of truth and
    // metric_dict.format_for_prompt() doesn't read those fields anyway.
    const payload = confirmed.map((m) => ({
      name: m.name,
      description: m.description,
      sql_formula: m.sql_formula,
      tables: [],
      columns: [],
    }));

    try {
      const res = await confirmMetrics(payload);
      if (res.ok) {
        setMetrics(payload);
        setMetricsConfirmed(payload.length);
      }
    } catch {
      // Confirmation failure is non-blocking.
    }
    setScreen("chat");
  }

  function handleSkipMetrics() {
    setScreen("chat");
  }

  // ── Manage existing metrics — entered from Settings icon in chat ──────────

  async function handleOpenManageMetrics() {
    try {
      // Fetch current confirmed metrics from backend so we always show fresh
      // state (in case anything changed outside this session).
      const res = await getMetrics();
      if (res.ok && res.metrics) {
        const editable = res.metrics.map((m) => ({
          name: m.name,
          description: m.description,
          sql_formula: m.sql_formula,
        }));
        setProposedMetrics(editable);
      }
    } catch {
      // If fetch fails, fall back to whatever's in the local store
      const fallback = useMetricsStore.getState().metrics.map((m) => ({
        name: m.name,
        description: m.description,
        sql_formula: m.sql_formula,
      }));
      setProposedMetrics(fallback);
    }
    setScreen("metric_manage");
  }

  function handleCancelManageMetrics() {
    setScreen("chat");
  }

  // ── Question submission flow ──────────────────────────────────────────────

  async function handleSubmitQuestion(question: string) {
    const turnId = crypto.randomUUID();
    const submittedAt = new Date();

    try {
      // run() drives the PipelineStatus stages in real time via SSE events
      // and resolves with the final QueryResponse when the stream completes.
      const res = await pipeline.run(question);

      if (!res || !res.ok) {
        const errorTurn: ConversationTurn = {
          id: turnId,
          userQuestion: question,
          timestamp: submittedAt,
          response: {
            narration:
              res?.error ??
              "Something went wrong while running this question. Please try again.",
            metric: null,
            chart: null,
            sql: "",
            corrections: [],
            masked_columns: [],
          },
        };
        setTurns((prev) => [...prev, errorTurn]);
        pipeline.reset();
        return;
      }

      // Hold the all-complete state for a meaningful beat so the user
      // can read the resolved details before PipelineStatus unmounts.
      await new Promise((r) => setTimeout(r, 600));

      const responseData = queryResponseToResponseData(res);
      const turn: ConversationTurn = {
        id: turnId,
        userQuestion: question,
        timestamp: submittedAt,
        response: responseData,
      };
      setTurns((prev) => [...prev, turn]);
      pipeline.reset();
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      const errorTurn: ConversationTurn = {
        id: turnId,
        userQuestion: question,
        timestamp: submittedAt,
        response: {
          narration: `Error: ${msg}`,
          metric: null,
          chart: null,
          sql: "",
          corrections: [],
          masked_columns: [],
        },
      };
      setTurns((prev) => [...prev, errorTurn]);
      pipeline.reset();
    }
  }

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <AppShell
      onWordmarkClick={() => {}}
      onHistoryClick={() => {}}
      onSettingsClick={
        screen === "chat" ? handleOpenManageMetrics : undefined
      }
      onThemeToggle={() => {}}
    >
      {screen === "boot" && <BootScreen />}
      {screen === "boot_error" && <BootErrorScreen message={bootError} />}
      {screen === "metric_setup" && (
        <MetricScreen
          proposedMetrics={proposedMetrics}
          mode="first_run"
          onConfirm={handleConfirmMetrics}
          onSkip={handleSkipMetrics}
        />
      )}
      {screen === "metric_manage" && (
        <MetricScreen
          proposedMetrics={proposedMetrics}
          mode="manage"
          onConfirm={handleConfirmMetrics}
          onSkip={handleCancelManageMetrics}
        />
      )}
      {screen === "chat" && (
        <ConversationView
          turns={turns}
          status={pipeline.stages}
          onSubmitQuestion={handleSubmitQuestion}
        />
      )}
    </AppShell>
  );
}

// ─── BootScreen ──────────────────────────────────────────────────────────────

function BootScreen() {
  return (
    <div className="h-[calc(100vh-65px)] flex flex-col items-center justify-center gap-4">
      <Loader2 className="size-5 text-brand animate-spin" strokeWidth={2.5} />
      <p className="font-sans text-sm text-muted-foreground">
        Initialising your data workspace
      </p>
    </div>
  );
}

// ─── BootErrorScreen ─────────────────────────────────────────────────────────

function BootErrorScreen({ message }: { message: string | null }) {
  return (
    <div className="h-[calc(100vh-65px)] flex flex-col items-center justify-center gap-3 px-6">
      <p className="font-sans text-base font-medium text-foreground text-center text-balance max-w-md">
        Couldn't connect to the data workspace
      </p>
      <p className="font-sans text-sm text-muted-foreground text-center text-balance max-w-md">
        {message ?? "Check that the Flask backend is running on port 5000."}
      </p>
    </div>
  );
}

export default App;