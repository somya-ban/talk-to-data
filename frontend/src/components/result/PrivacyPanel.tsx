'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { ChevronRight } from 'lucide-react'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible'
import { cn } from '@/lib/utils'

// ─── Types ────────────────────────────────────────────────────────────────────

export interface MaskedColumn {
  /** The column name as it appears in the schema */
  column: string
  /** The PII pattern that triggered masking (e.g. "name", "email", "phone") */
  pattern: string
}

export interface PrivacyPanelProps {
  /** The list of masked columns. If empty or undefined, component renders nothing. */
  maskedColumns: MaskedColumn[]
  /** Initial open state. Default: false */
  defaultOpen?: boolean
  /** Extra classes for the outermost wrapper */
  className?: string
}

// ─── Component ────────────────────────────────────────────────────────────────

export function PrivacyPanel({
  maskedColumns,
  defaultOpen = false,
  className,
}: PrivacyPanelProps) {
  const [open, setOpen] = useState(defaultOpen)

  if (!maskedColumns || maskedColumns.length === 0) {
    return null
  }

  const count = maskedColumns.length
  const countLabel = `${count} ${count === 1 ? 'column' : 'columns'} masked`

  return (
    <Collapsible
      open={open}
      onOpenChange={setOpen}
      className={cn('group', className)}
    >
      <CollapsibleTrigger asChild>
        <button
          type="button"
          className={cn(
            'audit-trigger inline-flex items-center gap-2 py-2 px-3',
            'cursor-pointer select-none',
            'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
          )}
          aria-label={`${open ? 'Collapse' : 'Expand'} privacy details (${countLabel})`}
        >
          <span
            className="size-1.5 rounded-full bg-success shrink-0"
            aria-hidden="true"
          />

          <motion.span
            animate={{ rotate: open ? 90 : 0 }}
            transition={{ duration: 0.2, ease: 'easeOut' }}
            className="flex items-center"
            aria-hidden="true"
          >
            <ChevronRight className="size-3.5 text-muted-foreground group-hover:text-foreground transition-colors duration-150" />
          </motion.span>

          <span
            className={cn(
              'font-sans text-[13px] font-semibold text-foreground/80',
              'transition-colors duration-150 group-hover:text-foreground',
            )}
          >
            Privacy
          </span>

          <span className="font-mono text-xs text-muted-foreground/60 group-hover:text-muted-foreground transition-colors duration-150">
            ·&nbsp;{countLabel}
          </span>
        </button>
      </CollapsibleTrigger>

      <CollapsibleContent>
        <div className="mt-2 flex flex-col gap-2.5">
          {maskedColumns.map((item) => (
            <div
              key={item.column}
              className="flex items-center gap-3"
            >
              <span className="font-mono text-xs font-medium text-foreground/85">
                {item.column}
              </span>

              <span
                className="text-muted-foreground/60 font-mono text-xs"
                aria-hidden="true"
              >
                →
              </span>

              <span className="font-mono text-xs text-muted-foreground">
                masked
              </span>
            </div>
          ))}
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}

// ─── v0 preview wrapper ──────────────────────────────────────────────────────

export default function PrivacyPanelPreview() {
  const sampleColumns: MaskedColumn[] = [
    { column: 'Customer.name', pattern: 'name' },
    { column: 'Customer.email', pattern: 'email' },
    { column: 'Account.account_number', pattern: 'account_number' },
  ]

  return (
    <div className="min-h-screen bg-background flex items-start justify-center pt-24 px-6">
      <div className="w-full max-w-2xl">
        <PrivacyPanel maskedColumns={sampleColumns} defaultOpen={true} />
      </div>
    </div>
  )
}