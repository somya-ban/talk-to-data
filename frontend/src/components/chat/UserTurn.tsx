'use client'

import { cn } from '@/lib/utils'

// ─── Types ────────────────────────────────────────────────────────────────────

export interface UserTurnProps {
  /** The question text the user submitted */
  question: string
  /** When the question was submitted. Accepts Date or ISO string. */
  timestamp: Date | string
  /** Extra classes for the outer wrapper */
  className?: string
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

const formatter = new Intl.DateTimeFormat('en-GB', {
  hour: '2-digit',
  minute: '2-digit',
  hour12: false,
})

function formatTimestamp(timestamp: Date | string): string {
  const date = timestamp instanceof Date ? timestamp : new Date(timestamp)
  return formatter.format(date)
}

// ─── Component ───────────────────────────────────────────────────────────────

export function UserTurn({ question, timestamp, className }: UserTurnProps) {
  if (!question) return null

  return (
    <div className={cn('w-full', className)}>
      <p
        className={cn(
          'font-sans',
          'text-base',
          'font-medium',
          'text-foreground/90',
          'leading-snug',
          'tracking-tight',
        )}
      >
        <span
          className={cn(
            'float-right',
            'ml-3',
            'font-mono',
            'text-xs',
            'font-normal',
            'text-muted-foreground/50',
            'tabular-nums',
            'whitespace-nowrap',
            'pt-1',
          )}
          aria-label={`Submitted at ${formatTimestamp(timestamp)}`}
        >
          {formatTimestamp(timestamp)}
        </span>
        {question}
      </p>
    </div>
  )
}

// ─── v0 preview wrapper ──────────────────────────────────────────────────────

export default function UserTurnPreview() {
  return (
    <div className="min-h-screen bg-background pt-24 px-6">
      <div className="max-w-2xl mx-auto w-full flex flex-col gap-8">
        <UserTurn
          question="What's our total balance across all customer accounts?"
          timestamp={new Date('2026-04-30T14:32:00')}
        />
        <UserTurn
          question="Show me the top 10 customers by trade volume in Q1, broken down by region. I want to understand if our high-value clients are concentrated in any particular geography."
          timestamp={new Date('2026-04-30T14:35:00')}
        />
        <UserTurn
          question="How many high-risk customers do we have?"
          timestamp={new Date('2026-04-30T14:38:00')}
        />
      </div>
    </div>
  )
}