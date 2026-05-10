'use client';

import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowRight, Plus } from "lucide-react";
import { cn } from "@/lib/utils";
import { MetricCard } from "@/components/metrics/MetricCard";
import { MetricEditor } from "@/components/metrics/MetricEditor";
import type { MetricDefinition } from "@/components/metrics/MetricCard";

// ─── Types ────────────────────────────────────────────────────────────────────

export type MetricScreenMode = "first_run" | "manage";

export interface MetricScreenProps {
  /** The metrics to present. In first_run, these are LLM proposals. In manage, these are confirmed metrics from storage. */
  proposedMetrics: MetricDefinition[];
  /**
   * What surface the user reached this screen from.
   * - "first_run": initial setup, primary action is Confirm, secondary is Skip
   * - "manage":    later return visit, primary action is Save changes, secondary is Cancel
   */
  mode?: MetricScreenMode;
  /** Called with the final confirmed (non-deleted) metrics list when the primary action is clicked. */
  onConfirm: (confirmedMetrics: MetricDefinition[]) => void;
  /** Called when Skip (first_run) or Cancel (manage) is clicked. */
  onSkip: () => void;
  /** Extra classes for the outer wrapper */
  className?: string;
}

// ─── Animation timing — quick crossfade, signals the swap without drama ──────

const fadeTransition = { duration: 0.15, ease: "easeOut" as const };

const EMPTY_METRIC: MetricDefinition = {
  name: "",
  description: "",
  sql_formula: "",
};

// ─── Component ────────────────────────────────────────────────────────────────

export function MetricScreen({
  proposedMetrics,
  mode = "first_run",
  onConfirm,
  onSkip,
  className,
}: MetricScreenProps) {
  const [metrics, setMetrics] = useState<MetricDefinition[]>(proposedMetrics);
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [deletedIndexes, setDeletedIndexes] = useState<Set<number>>(new Set());
  // When true, an empty MetricEditor is appended to the list for new-metric creation
  const [isCreatingNew, setIsCreatingNew] = useState(false);

  function handleEdit(index: number) {
    setEditingIndex(index);
  }

  function handleSave(updated: MetricDefinition) {
    if (editingIndex === null) return;
    setMetrics((prev) => {
      const next = [...prev];
      next[editingIndex] = updated;
      return next;
    });
    setEditingIndex(null);
  }

  function handleCancel() {
    setEditingIndex(null);
  }

  function handleDelete(index: number, deleted: boolean): void {
    setDeletedIndexes((prev) => {
      const next = new Set(prev);
      if (deleted) next.add(index);
      else next.delete(index);
      return next;
    });
  }

  function handleStartCreating() {
    setIsCreatingNew(true);
  }

  function handleSaveNew(created: MetricDefinition) {
    if (!created.name.trim()) {
      // Don't save metrics with empty names — quietly cancel
      setIsCreatingNew(false);
      return;
    }
    setMetrics((prev) => [...prev, created]);
    setIsCreatingNew(false);
  }

  function handleCancelNew() {
    setIsCreatingNew(false);
  }

  function handleConfirm() {
    const confirmed = metrics.filter((_, i) => !deletedIndexes.has(i));
    onConfirm(confirmed);
  }

  const totalCount = metrics.length;
  const activeCount = metrics.filter((_, i) => !deletedIndexes.has(i)).length;
  const noneRemain = activeCount === 0;
  const hasDeletions = deletedIndexes.size > 0;

  // ── Mode-dependent copy ──────────────────────────────────────────────────
  const headline =
    mode === "first_run"
      ? "Confirm your business metrics"
      : "Manage your business metrics";

  const subline =
    mode === "first_run"
      ? "These definitions were inferred from your schema. Review and edit any that don't match how your team thinks about the data."
      : "Edit existing definitions, add new metrics, or remove ones you no longer use. Every query is grounded against this list.";

  const primaryLabel =
    mode === "first_run"
      ? noneRemain
        ? "Continue without metrics"
        : "Confirm metrics"
      : "Save changes";

  const secondaryLabel = mode === "first_run" ? "Skip" : "Cancel";

  const statusText = noneRemain
    ? `0 of ${totalCount} metrics will be saved`
    : hasDeletions
      ? `${activeCount} of ${totalCount} metrics will be saved`
      : `${activeCount} ${activeCount === 1 ? "metric" : "metrics"} will be saved`;

  return (
    <div
      className={cn(
        "h-full w-full bg-transparent flex flex-col",
        className,
      )}
    >
      {/* Zone 1: Header */}
      <header className="flex flex-col items-center gap-4 max-w-2xl mx-auto w-full px-6 pt-16 pb-10">
        <h1 className="font-sans text-4xl md:text-5xl font-medium tracking-tight text-balance text-center bg-gradient-to-b from-foreground to-foreground/70 bg-clip-text text-transparent">
          {headline}
        </h1>
        <p className="font-sans text-base text-muted-foreground leading-relaxed text-center max-w-lg mx-auto">
          {subline}
        </p>
      </header>

      {/* Zone 2: Scrollable body */}
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-2xl mx-auto w-full px-6 py-6 flex flex-col gap-3">
          {metrics.length === 0 && !isCreatingNew ? (
            <div className="flex flex-col items-center gap-4 py-12">
              <p className="text-sm text-muted-foreground italic text-center">
                No metrics defined yet.
              </p>
              <button
                type="button"
                onClick={handleStartCreating}
                className="inline-flex items-center gap-1.5 font-sans text-sm font-medium text-brand hover:text-brand/80 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring rounded-sm transition-colors duration-150"
              >
                <Plus className="size-3.5" />
                Add your first metric
              </button>
            </div>
          ) : (
            <>
              {metrics.map((metric, index) => (
                <div key={`${metric.name}-${index}`}>
                  <AnimatePresence mode="wait">
                    {editingIndex === index ? (
                      <motion.div
                        key={`${metric.name}-editor`}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        transition={fadeTransition}
                      >
                        <MetricEditor
                          metric={metric}
                          onSave={handleSave}
                          onCancel={handleCancel}
                        />
                      </motion.div>
                    ) : (
                      <motion.div
                        key={`${metric.name}-card`}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        transition={fadeTransition}
                      >
                        <MetricCard
                          metric={metric}
                          isDeleted={deletedIndexes.has(index)}
                          onEdit={() => handleEdit(index)}
                          onDelete={(deleted: boolean) => handleDelete(index, deleted)}
                        />
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              ))}

              {/* New-metric editor — appears when "Add metric" is clicked */}
              <AnimatePresence>
                {isCreatingNew && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={fadeTransition}
                  >
                    <MetricEditor
                      metric={EMPTY_METRIC}
                      onSave={handleSaveNew}
                      onCancel={handleCancelNew}
                    />
                  </motion.div>
                )}
              </AnimatePresence>

              {/* "Add metric" entry — hidden while a new-metric editor is open */}
              {!isCreatingNew && (
                <button
                  type="button"
                  onClick={handleStartCreating}
                  className={cn(
                    "group w-full flex items-center justify-center gap-2",
                    "py-5 rounded-lg",
                    "border border-dashed border-border/60 hover:border-border",
                    "text-muted-foreground hover:text-foreground",
                    "transition-colors duration-150 ease-out",
                    "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
                  )}
                  aria-label="Add a new metric"
                >
                  <Plus className="size-4 shrink-0" />
                  <span className="font-sans text-sm font-medium">Add metric</span>
                </button>
              )}
            </>
          )}
        </div>
      </main>

      {/* Zone 3: Footer */}
      <footer className="border-t border-border/50 bg-background/80 backdrop-blur-md px-6 py-5">
        <div className="max-w-2xl mx-auto w-full flex flex-row items-center justify-between gap-4">
          <span className="font-sans text-sm text-muted-foreground">
            {statusText}
          </span>

          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={onSkip}
              className="font-sans text-sm font-medium text-muted-foreground hover:text-foreground bg-transparent px-4 py-2 rounded-md transition-colors duration-150 ease-out focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            >
              {secondaryLabel}
            </button>

            <button
              type="button"
              onClick={handleConfirm}
              className="inline-flex items-center gap-1.5 font-sans text-sm font-semibold bg-brand text-brand-foreground hover:bg-brand/90 pl-5 pr-4 py-2 rounded-md transition-colors duration-150 ease-out focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-brand focus-visible:ring-offset-2 focus-visible:ring-offset-background"
            >
              {primaryLabel}
              <ArrowRight className="size-3.5" aria-hidden="true" />
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}

// ─── v0 preview wrapper ──────────────────────────────────────────────────────

const SAMPLE_METRICS: MetricDefinition[] = [
  {
    name: "total_balance",
    description:
      "Sum of all account balances across all AccountBalance records.",
    sql_formula:
      "SELECT SUM(ab.balance) AS total_balance FROM AccountBalance ab",
  },
  {
    name: "active_customer_count",
    description:
      "Number of customers with at least one transaction in the past 90 days.",
    sql_formula:
      "SELECT COUNT(DISTINCT customer_id) AS active_customer_count FROM Transaction WHERE transaction_date >= CURRENT_DATE - INTERVAL 90 DAY",
  },
];

export default function MetricScreenPreview() {
  return (
    <MetricScreen
      proposedMetrics={SAMPLE_METRICS}
      onConfirm={(m) => console.log("Confirmed:", m)}
      onSkip={() => console.log("Skipped")}
    />
  );
}