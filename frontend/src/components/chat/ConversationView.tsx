'use client';

import { useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';

import { UserTurn } from '@/components/chat/UserTurn';
import { ResponseTurn } from '@/components/result/ResponseTurn';
import type { ResponseData } from '@/components/result/ResponseTurn';
import { PipelineStatus } from '@/components/chat/PipelineStatus';
import type { PipelineStage } from '@/components/chat/PipelineStatus';
import { Composer } from '@/components/chat/Composer';
import { EmptyState } from '@/components/chat/EmptyState';

// ─── Types ────────────────────────────────────────────────────────────────────

export interface ConversationTurn {
  /** Stable identifier for React keys and animations */
  id: string;
  /** The user's question text */
  userQuestion: string;
  /** When the question was submitted */
  timestamp: Date | string;
  /** The AI's response data */
  response: ResponseData;
  /** When true, ResponseTurn renders without typewriter animation. Used for historical replay. */
  skipAnimation?: boolean;
}

/**
 * The pipeline status while a question is being processed.
 * Activity is determined by each stage's own `status` field
 * ("pending" | "active" | "complete"); no separate active-index needed.
 */
export type PipelineStatusState = PipelineStage[];

export interface ConversationViewProps {
  /** All conversation turns, in chronological order. */
  turns: ConversationTurn[];
  /** Current pipeline stages while a question is being processed. null when idle. */
  status: PipelineStatusState | null;
  /** Called when the user submits a question via the Composer or a SuggestedChip. */
  onSubmitQuestion: (question: string) => void;
  /** Extra classes for the outer wrapper */
  className?: string;
}

// ─── Component ───────────────────────────────────────────────────────────────

export function ConversationView({
  turns,
  status,
  onSubmitQuestion,
  className,
}: ConversationViewProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const isProcessing = status !== null;

  // Scroll to bottom whenever a new turn is added
  useEffect(() => {
    if (turns.length === 0) return;
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
  }, [turns.length]);

  // Also scroll to bottom the moment processing starts (idle → active)
  useEffect(() => {
    if (!isProcessing) return;
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
  }, [isProcessing]);

  const isEmpty = turns.length === 0 && status === null;

  return (
    <main
      className={cn(
        'relative w-full bg-transparent flex flex-col h-[calc(100vh-65px)]',
        className,
      )}
    >
      {/* Zone 1: Scrollable conversation area */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        <div className="max-w-2xl mx-auto w-full px-6 py-12">
          {isEmpty ? (
            <div className="h-full flex items-center justify-center pb-32">
              <EmptyState onSelectQuestion={onSubmitQuestion} />
            </div>
          ) : (
            <div className="flex flex-col gap-16">
              {turns.map((turn) => (
                <motion.div
                  key={turn.id}
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.2, ease: 'easeOut' }}
                  layout
                  className="flex flex-col gap-6"
                >
                  <UserTurn
                    question={turn.userQuestion}
                    timestamp={turn.timestamp}
                  />
                  <ResponseTurn
                    response={turn.response}
                    skipAnimation={turn.skipAnimation ?? false}
                  />
                </motion.div>
              ))}

              {status !== null && (
                <motion.div
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.2, ease: 'easeOut' }}
                >
                  <PipelineStatus stages={status} />
                </motion.div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Zone 2: Sticky composer */}
      <div className="sticky bottom-0 bg-background relative">
        <div
          aria-hidden="true"
          className="absolute -top-6 left-0 h-6 w-full pointer-events-none bg-gradient-to-t from-background to-transparent"
        />
        <div className="max-w-2xl mx-auto w-full px-6 py-5">
          <Composer onSubmit={onSubmitQuestion} />
        </div>
      </div>
    </main>
  );
}

// ─── v0 preview wrapper ──────────────────────────────────────────────────────

const SAMPLE_TURNS: ConversationTurn[] = [
  {
    id: 'turn-1',
    userQuestion: "What's our total balance across all customer accounts?",
    timestamp: new Date('2026-04-30T14:32:00'),
    response: {
      narration:
        'The total balance across all customer accounts is £4.2 million, with 60% concentrated in the top 5 customers.',
      metric: {
        name: 'total_balance',
        description:
          'Sum of all account balances across all AccountBalance records',
        sql_formula:
          'SELECT SUM(ab.balance) AS total_balance FROM AccountBalance ab',
      },
      chart: {
        chart_type: 'bar',
        data: [
          { region: 'London', customers: 1240 },
          { region: 'Manchester', customers: 870 },
          { region: 'Edinburgh', customers: 560 },
        ],
        columns: ['region', 'customers'],
        x_key: 'region',
        y_key: 'customers',
      },
      sql: 'SELECT region, COUNT(*) FROM Customer GROUP BY region;',
      corrections: [],
      masked_columns: [],
    },
    skipAnimation: true,
  },
];

export default function ConversationViewPreview() {
  return (
    <ConversationView
      turns={SAMPLE_TURNS}
      status={null}
      onSubmitQuestion={(q) => console.log('Submit:', q)}
    />
  );
}