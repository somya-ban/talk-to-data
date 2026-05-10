"use client";

import { AnimatePresence, motion } from "framer-motion";
import { Check, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface PipelineStage {
  label: string;
  detail?: string | string[];
  status: "pending" | "active" | "complete";
}

export interface PipelineStatusProps {
  stages: PipelineStage[];
}

// ─── Sub-components ───────────────────────────────────────────────────────────

/**
 * Leading glyph: spinning Loader2 for active, Check for complete.
 * Loader2 is the canonical in-progress glyph used by Vercel, Linear,
 * and most modern AI products — legible at small sizes.
 */
function StageGlyph({ status }: { status: "active" | "complete" }) {
  return (
    <span className="relative size-4 shrink-0 flex items-center justify-center">
      <AnimatePresence mode="wait" initial={false}>
        {status === "active" ? (
          <motion.span
            key="loader"
            className="absolute inset-0 flex items-center justify-center"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
          >
            <Loader2 className="size-3.5 text-brand animate-spin" strokeWidth={2.5} />
          </motion.span>
        ) : (
          <motion.span
            key="check"
            className="absolute inset-0 flex items-center justify-center"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
          >
            <Check className="size-3.5 text-success" strokeWidth={2.5} />
          </motion.span>
        )}
      </AnimatePresence>
    </span>
  );
}

/**
 * Inline detail fragment — clearly subordinate to label.
 */
function StageDetail({
  detail,
  isActive,
}: {
  detail: string | string[];
  isActive: boolean;
}) {
  const rendered = Array.isArray(detail) ? detail.join(" · ") : detail;

  return (
    <>
      <span className="text-muted-foreground/30 text-sm select-none mx-1">
        ·
      </span>
      <span
        className={cn(
          "font-mono text-[13px] font-normal tracking-tight",
          isActive ? "text-muted-foreground" : "text-muted-foreground/55"
        )}
      >
        {rendered}
      </span>
    </>
  );
}

/**
 * A single pipeline stage row. The active state has spatial presence:
 * a soft brand-tinted background, a glowing left border, and a clearer
 * pulse on the label — so the user feels which stage is alive.
 */
function StageRow({ stage }: { stage: PipelineStage }) {
  if (stage.status === "pending") return null;

  const isActive = stage.status === "active";

  return (
    <motion.div
      className="relative flex flex-row items-center gap-3 leading-none py-1"
      initial={{ opacity: 0, x: -6 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -4 }}
      transition={{ duration: 0.32, ease: "easeOut" }}
    >
      <StageGlyph status={isActive ? "active" : "complete"} />

      <motion.span
        className={cn(
          "font-sans text-sm tracking-tight",
          isActive
            ? "text-foreground font-semibold"
            : "text-foreground/85 font-medium"
        )}
        animate={
          isActive
            ? { opacity: [1.0, 0.7, 1.0] }
            : { opacity: 1 }
        }
        transition={
          isActive
            ? {
                duration: 1.8,
                ease: "easeInOut",
                repeat: Infinity,
                repeatType: "loop",
              }
            : { duration: 0.28, ease: "easeOut" }
        }
      >
        {stage.label}
      </motion.span>

      {stage.detail !== undefined && stage.detail !== null && (
        <StageDetail detail={stage.detail} isActive={isActive} />
      )}
    </motion.div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

export function PipelineStatus({ stages }: PipelineStatusProps) {
  const visible = stages.filter((s) => s.status !== "pending");

  return (
    <div className="card-surface px-5 py-4 max-w-2xl mx-auto">
      <div className="flex flex-col gap-2.5">
        <AnimatePresence initial={false}>
          {visible.map((stage, i) => (
            <StageRow key={`${stage.label}-${i}`} stage={stage} />
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}

// ─── Preview wrapper (v0 only — strip when pasting into project) ──────────────

export default function PipelineStatusPreview() {
  const stages: PipelineStage[] = [
    {
      label: "Finding the right tables",
      detail: ["Customer", "Account", "Trade"],
      status: "complete",
    },
    {
      label: "Composing the query",
      detail: "Using your definition of total_balance",
      status: "complete",
    },
    {
      label: "Running the query",
      status: "active",
    },
    {
      label: "Summarising the results",
      status: "pending",
    },
  ];

  return (
    <div className="dark min-h-screen bg-background flex items-start justify-center pt-24 px-6">
      <PipelineStatus stages={stages} />
    </div>
  );
}