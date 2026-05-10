/**
 * Pipeline status — initialisation state, readiness, current stage.
 *
 * Read from: App.tsx (to decide which screen to show),
 *            PipelineStatus.tsx (to render the active stage).
 * Written by: api.ts call sites in App.tsx and Composer.tsx.
 */

import { create } from "zustand";

export type PipelineStage =
  | "idle"
  | "linking"
  | "generating"
  | "executing"
  | "narrating";

interface StatusState {
  initialised: boolean;
  ready: boolean;
  metricsConfirmed: number;
  qaPairs: number;
  stage: PipelineStage;
  setInitialised: (v: boolean) => void;
  setReady: (v: boolean) => void;
  setMetricsConfirmed: (n: number) => void;
  setQaPairs: (n: number) => void;
  setStage: (s: PipelineStage) => void;
  reset: () => void;
}

export const useStatusStore = create<StatusState>((set) => ({
  initialised: false,
  ready: false,
  metricsConfirmed: 0,
  qaPairs: 0,
  stage: "idle",
  setInitialised: (v) => set({ initialised: v }),
  setReady: (v) => set({ ready: v }),
  setMetricsConfirmed: (n) => set({ metricsConfirmed: n }),
  setQaPairs: (n) => set({ qaPairs: n }),
  setStage: (s) => set({ stage: s }),
  reset: () =>
    set({
      initialised: false,
      ready: false,
      metricsConfirmed: 0,
      qaPairs: 0,
      stage: "idle",
    }),
}));