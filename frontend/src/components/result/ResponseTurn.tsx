"use client";

import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

import { Narration } from "@/components/result/Narration";
import { MetricAttribution } from "@/components/result/MetricAttribution";
import { ChartCanvas } from "@/components/result/ChartCanvas";
import type { ChartType } from "@/components/result/ChartCanvas";
import { SqlPanel } from "@/components/result/SqlPanel";
import { CorrectionPanel } from "@/components/result/CorrectionPanel";
import type { CorrectionEntry } from "@/components/result/CorrectionPanel";
import { PrivacyPanel } from "@/components/result/PrivacyPanel";
import type { MaskedColumn } from "@/components/result/PrivacyPanel";

export type { ChartType, CorrectionEntry, MaskedColumn };

// ─── Types ────────────────────────────────────────────────────────────────────

export interface ResponseData {
  narration: string;
  metric: {
    name: string;
    description: string;
    sql_formula: string;
  } | null;
  chart: {
    chart_type: ChartType;
    data: Array<Record<string, string | number | boolean | null>>;
    columns: string[];
    x_key?: string;
    y_key?: string;
  } | null;
  sql: string;
  corrections: CorrectionEntry[];
  masked_columns: MaskedColumn[];
}

export interface ResponseTurnProps {
  response: ResponseData;
  /**
   * When true, skip the typewriter animation and reveal everything immediately.
   * Use when re-rendering a historical response.
   * @default false
   */
  skipAnimation?: boolean;
  /** Extra classes for the outermost wrapper */
  className?: string;
}

// ─── Cascade stages ───────────────────────────────────────────────────────────

type RevealStage = "narrating" | "attribution" | "chart_and_panels";

const fadeUp = {
  hidden: { opacity: 0, y: 4 },
  visible: { opacity: 1, y: 0 },
};

// ─── Component ────────────────────────────────────────────────────────────────

export function ResponseTurn({
  response,
  skipAnimation = false,
  className,
}: ResponseTurnProps) {
  const [revealStage, setRevealStage] = useState<RevealStage>(
    skipAnimation ? "chart_and_panels" : "narrating"
  );

  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  const handleNarrationComplete = () => {
    if (skipAnimation) return;
    setRevealStage("attribution");
    timerRef.current = setTimeout(() => {
      setRevealStage("chart_and_panels");
    }, 200);
  };

  const showMetric = response.metric !== null;
  const showChart =
    response.chart !== null && response.chart.chart_type != null;
  const showSql = response.sql.trim().length > 0;
  const showCorrections = response.corrections.length > 0;
  const showPrivacy = response.masked_columns.length > 0;
  const showAuditGroup = showSql || showCorrections || showPrivacy;

  const isAttributionVisible =
    revealStage === "attribution" || revealStage === "chart_and_panels";
  const isChartAndPanelsVisible = revealStage === "chart_and_panels";

  return (
    <div
      className={cn(
        "flex flex-col max-w-2xl mx-auto w-full",
        className
      )}
      role="region"
      aria-label="Response"
    >
      {/* 1. Narration — primary answer */}
      {response.narration && (
        <Narration
          text={response.narration}
          enabled={!skipAnimation}
          onComplete={handleNarrationComplete}
        />
      )}

      {/* 2. MetricAttribution — sits close to narration as its byline */}
      {showMetric && (
        <motion.div
          initial="hidden"
          animate={isAttributionVisible ? "visible" : "hidden"}
          variants={fadeUp}
          transition={{ duration: 0.25, ease: "easeOut" }}
          className="mt-3"
        >
          <MetricAttribution metric={response.metric} />
        </motion.div>
      )}

      {/* 3. ChartCanvas — the visualization, with breathing space above */}
      {showChart && response.chart && (
        <motion.div
          initial="hidden"
          animate={isChartAndPanelsVisible ? "visible" : "hidden"}
          variants={fadeUp}
          transition={{ duration: 0.3, ease: "easeOut", delay: 0.1 }}
          className="mt-8"
        >
          <ChartCanvas
            chartType={response.chart.chart_type}
            data={response.chart.data}
            columns={response.chart.columns}
            xKey={response.chart.x_key}
            yKey={response.chart.y_key}
            metricName={response.metric?.name}
          />
        </motion.div>
      )}

      {/* 4. Audit group — three panels (no divider; spacing carries the transition) */}
      {showAuditGroup && (
        <motion.div
          initial="hidden"
          animate={isChartAndPanelsVisible ? "visible" : "hidden"}
          variants={fadeUp}
          transition={{ duration: 0.3, ease: "easeOut", delay: 0.1 }}
          className="mt-12 flex flex-col gap-1"
        >
          {showSql && <SqlPanel sql={response.sql} defaultOpen={false} />}

          {showCorrections && (
            <CorrectionPanel
              corrections={response.corrections}
              defaultOpen={false}
            />
          )}

          {showPrivacy && (
            <PrivacyPanel
              maskedColumns={response.masked_columns}
              defaultOpen={false}
            />
          )}
        </motion.div>
      )}
    </div>
  );
}

// ─── v0 preview wrapper ──────────────────────────────────────────────────────

export default function ResponseTurnPreview() {
  const sample: ResponseData = {
    narration:
      "The total balance across all customer accounts is £4.2 million, with 60% concentrated in the top 5 customers. Average account balance sits at £104,000 — a useful benchmark when evaluating individual exposures.",
    metric: {
      name: "total_balance",
      description:
        "Sum of all account balances across all AccountBalance records",
      sql_formula:
        "SELECT SUM(ab.balance) AS total_balance FROM AccountBalance ab",
    },
    chart: {
      chart_type: "bar",
      data: [
        { region: "London", customers: 1240 },
        { region: "Manchester", customers: 870 },
        { region: "Edinburgh", customers: 560 },
        { region: "Bristol", customers: 420 },
        { region: "Cardiff", customers: 310 },
      ],
      columns: ["region", "customers"],
      x_key: "region",
      y_key: "customers",
    },
    sql: `SELECT
  region,
  COUNT(DISTINCT customer_id) AS customers
FROM Customer
GROUP BY region
ORDER BY customers DESC;`,
    corrections: [
      {
        attempt: 1,
        sql: "SELECT first_name FROM Customer GROUP BY region;",
        error_type: "COLUMN_NOT_FOUND",
        error_message:
          "Column 'first_name' does not exist on table Customer.",
      },
    ],
    masked_columns: [
      { column: "Customer.name", pattern: "name" },
      { column: "Customer.email", pattern: "email" },
    ],
  };

  return (
    <div className="min-h-screen bg-background pt-20 pb-32 px-6">
      <ResponseTurn response={sample} />
    </div>
  );
}