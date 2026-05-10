/**
 * Confirmed metrics — loaded at startup, used by the metric proposal flow,
 * referenced by Composer to suggest follow-up questions.
 */

import { create } from "zustand";
import type { Metric, MetricProposal } from "@/lib/types";

interface MetricsState {
  metrics: Metric[];
  proposals: MetricProposal[];
  setMetrics: (m: Metric[]) => void;
  setProposals: (p: MetricProposal[]) => void;
  clearProposals: () => void;
  isEmpty: () => boolean;
}

export const useMetricsStore = create<MetricsState>((set, get) => ({
  metrics: [],
  proposals: [],
  setMetrics: (m) => set({ metrics: m }),
  setProposals: (p) => set({ proposals: p }),
  clearProposals: () => set({ proposals: [] }),
  isEmpty: () => get().metrics.length === 0,
}));