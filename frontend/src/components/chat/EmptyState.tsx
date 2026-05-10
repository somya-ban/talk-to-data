'use client';

import { BarChart3, Users, TrendingUp, Layers } from 'lucide-react';
import { cn } from '@/lib/utils';
import { SuggestedChips } from '@/components/chat/SuggestedChips';
import type { SuggestedChip } from '@/components/chat/SuggestedChips';

export type { SuggestedChip };

// ─── Default chips ──────────────────────────────────────────────────────────

const DEFAULT_CHIPS: SuggestedChip[] = [
  {
    question: "What's our total balance across all customer accounts?",
    label: 'Total balance across customer accounts',
    icon: BarChart3,
  },
  {
    question: 'Show me the top 10 customers by trade volume in Q1 this year',
    label: 'Top 10 customers by trade volume',
    icon: Users,
  },
  {
    question: 'How has cash inflow changed over the last 6 months?',
    label: 'Cash inflow trend over 6 months',
    icon: TrendingUp,
  },
  {
    question: 'Compare account balances by region',
    label: 'Account balances by region',
    icon: Layers,
  },
];

// ─── Types ───────────────────────────────────────────────────────────────────

export interface EmptyStateProps {
  /** Called when a suggested chip is clicked, with the full question text. */
  onSelectQuestion: (question: string) => void;
  /** Custom suggested chips. If not provided, uses banking defaults. */
  chips?: SuggestedChip[];
  /** Extra classes for the outer wrapper. */
  className?: string;
}

// ─── Component ───────────────────────────────────────────────────────────────

export function EmptyState({ onSelectQuestion, chips, className }: EmptyStateProps) {
  const resolvedChips = chips ?? DEFAULT_CHIPS;

  return (
    <div
      className={cn(
        'flex flex-col items-center gap-10',
        'max-w-2xl mx-auto w-full px-6',
        className,
      )}
    >
      <h1
        className={cn(
          'font-sans text-4xl md:text-5xl font-medium tracking-tight',
          'text-foreground text-center text-balance',
          'bg-gradient-to-b from-foreground to-foreground/70 bg-clip-text text-transparent',
        )}
      >
        Ask anything about your data
      </h1>

      <SuggestedChips
        chips={resolvedChips}
        onSelect={onSelectQuestion}
      />
    </div>
  );
}

// ─── v0 preview wrapper ──────────────────────────────────────────────────────

export default function EmptyStatePreview() {
  return (
    <div className="min-h-screen bg-background flex items-center justify-center px-6">
      <EmptyState onSelectQuestion={(q) => console.log('Selected:', q)} />
    </div>
  );
}