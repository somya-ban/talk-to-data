'use client'

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'

// ─── Types ────────────────────────────────────────────────────────────────────

export interface MetricAttributionProps {
  /** The confirmed metric being attributed. If null or undefined, renders nothing. */
  metric:
    | {
        name: string
        description: string
        sql_formula: string
      }
    | null
    | undefined
  /** Extra classes for the outer wrapper. */
  className?: string
}

// ─── Component ────────────────────────────────────────────────────────────────

export function MetricAttribution({ metric, className }: MetricAttributionProps) {
  if (!metric) return null

  return (
    <div className={cn('flex flex-row items-baseline flex-wrap', className)}>
      <span className="text-sm font-sans font-normal text-muted-foreground/60 select-none">
        Using:&nbsp;
      </span>

      <Tooltip>
        <TooltipTrigger asChild>
          <span
            className={cn(
              'text-sm font-sans font-medium text-brand',
              'cursor-help',
              'underline decoration-transparent underline-offset-[3px]',
              'hover:decoration-brand/50',
              'transition-[text-decoration-color] duration-200 ease-out',
            )}
          >
            {metric.name}
          </span>
        </TooltipTrigger>
        <TooltipContent
        side="top"
        align="start"
        sideOffset={6}
        className="max-w-md px-3 py-2"
        >
        <p className="font-mono text-[11px] text-muted-foreground mb-1 select-none tracking-wide uppercase">
            Formula
        </p>
        <p className="font-mono text-xs text-foreground break-words whitespace-pre-wrap leading-relaxed">
            {metric.sql_formula}
        </p>
        </TooltipContent>
      </Tooltip>

      <span
        className="text-sm text-muted-foreground/30 select-none"
        aria-hidden="true"
      >
        &nbsp;·&nbsp;
      </span>

      <span className="text-sm font-sans font-normal text-muted-foreground">
        {metric.description}
      </span>
    </div>
  )
}

// ─── v0 preview wrapper ───────────────────────────────────────────────────────

export default function MetricAttributionPreview() {
  return (
    <div className="dark min-h-screen bg-background flex items-start justify-center pt-32 px-6">
      <MetricAttribution
        metric={{
          name: 'total_balance',
          description:
            'Sum of all account balances across all AccountBalance records',
          sql_formula:
            'SELECT SUM(ab.balance) AS total_balance\nFROM AccountBalance ab',
        }}
      />
    </div>
  )
}