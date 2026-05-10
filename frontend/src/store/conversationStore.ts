/**
 * Conversation state — the list of turns currently rendered.
 *
 * UserTurn captures what the user typed.
 * PendingTurn is shown while a query is in flight (renders PipelineStatus).
 * AssistantTurn replaces the PendingTurn once the response arrives.
 */

import { create } from "zustand";
import type { Turn, QueryResponse, PendingTurn} from "@/lib/types";

interface ConversationState {
  turns: Turn[];
  addUserTurn: (question: string) => string;
  addPendingTurn: (id?: string) => string;
  updatePendingStage: (id: string, stage: PendingTurn["stage"]) => void;
  resolvePendingTurn: (id: string, response: QueryResponse) => void;
  clear: () => void;
}

function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

export const useConversationStore = create<ConversationState>((set) => ({
  turns: [],
  addUserTurn: (question) => {
    const id = generateId();
    set((state) => ({
      turns: [
        ...state.turns,
        { id, role: "user", question, timestamp: new Date().toISOString() },
      ],
    }));
    return id;
  },
  addPendingTurn: (id) => {
    const turnId = id ?? generateId();
    set((state) => ({
      turns: [
        ...state.turns,
        { id: turnId, role: "assistant", pending: true, stage: "linking" },
      ],
    }));
    return turnId;
  },
  updatePendingStage: (id, stage) => {
    set((state) => ({
      turns: state.turns.map((t) =>
        t.id === id && "pending" in t ? { ...t, stage } : t
      ),
    }));
  },
  resolvePendingTurn: (id, response) => {
    set((state) => ({
      turns: state.turns.map((t) =>
        t.id === id ? { id, role: "assistant", response } : t
      ),
    }));
  },
  clear: () => set({ turns: [] }),
}));