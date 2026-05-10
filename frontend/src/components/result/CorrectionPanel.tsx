'use client'

import { useState } from 'react'
import { ChevronRight } from 'lucide-react'
import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible'

// ─── Types ────────────────────────────────────────────────────────────────────

export interface CorrectionEntry {
  attempt: number
  sql: string
  error_type: string
  error_message: string
}

export interface CorrectionPanelProps {
  /** The list of prior failed attempts. If empty or undefined, component renders nothing. */
  corrections: CorrectionEntry[]
  /** Initial open state. Default: false */
  defaultOpen?: boolean
  /** Extra classes for the outermost wrapper */
  className?: string
}

// ─── SQL formatter for failed attempts ──────────────────────────────────────

/**
 * Pretty-print failed SQL onto multiple lines along major clause boundaries.
 * Same rules as SqlPanel's formatter — kept inline (not exported) so each
 * audit panel owns its own formatting concerns.
 */
function formatFailedSQL(sql: string): string {
  if (sql.includes('\n')) return sql;

  let s = sql.trim();

  const strings: string[] = [];
  s = s.replace(/'(?:[^']|'')*'/g, (match) => {
    strings.push(match);
    return `\x00${strings.length - 1}\x00`;
  });

  s = s.replace(
    /\s+(GROUP\s+BY|ORDER\s+BY|UNION\s+ALL|FROM|WHERE|HAVING|LIMIT|OFFSET|UNION)\s+/gi,
    (_, kw: string) => '\n' + kw.replace(/\s+/g, ' ').toUpperCase() + ' ',
  );

  s = s.replace(
    /\s+(LEFT\s+OUTER\s+JOIN|RIGHT\s+OUTER\s+JOIN|FULL\s+OUTER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|INNER\s+JOIN|CROSS\s+JOIN|JOIN)\s+/gi,
    (_, kw: string) => '\n' + kw.replace(/\s+/g, ' ').toUpperCase() + ' ',
  );

  s = s.replace(
    /\s+(AND|OR)\s+/gi,
    (_, kw: string) => '\n  ' + kw.toUpperCase() + ' ',
  );

  s = s.replace(/\x00(\d+)\x00/g, (_, idx: string) => strings[parseInt(idx, 10)]);

  return s;
}

// ─── Syntax highlighting (shared with SqlPanel) ─────────────────────────────

const SQL_KEYWORDS = new Set([
  'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'OUTER',
  'ON', 'AS', 'GROUP', 'BY', 'ORDER', 'HAVING', 'LIMIT', 'OFFSET',
  'AND', 'OR', 'NOT', 'IN', 'LIKE', 'BETWEEN', 'IS', 'NULL', 'CASE',
  'WHEN', 'THEN', 'ELSE', 'END', 'DISTINCT', 'UNION', 'ALL', 'WITH',
  'SUM', 'COUNT', 'AVG', 'MIN', 'MAX', 'DESC', 'ASC',
]);

function highlightFailedSQL(text: string): React.ReactNode[] {
  const tokenRe = /('(?:[^']|'')*')|(\b\d+(?:\.\d+)?\b)|([A-Za-z_]\w*)|(\s+)|([^\s])/g;
  const parts: React.ReactNode[] = [];
  let match: RegExpExecArray | null;
  let key = 0;

  while ((match = tokenRe.exec(text)) !== null) {
    const [, str, num, ident, ws, punct] = match;
    if (str) {
      parts.push(<span key={key++} className="text-foreground/60 italic">{str}</span>);
    } else if (num) {
      parts.push(<span key={key++} className="text-foreground/60">{num}</span>);
    } else if (ident) {
      const upper = ident.toUpperCase();
      if (SQL_KEYWORDS.has(upper)) {
        parts.push(<span key={key++} className="text-foreground font-semibold">{ident}</span>);
      } else {
        parts.push(<span key={key++} className="text-foreground/75">{ident}</span>);
      }
    } else if (ws) {
      parts.push(<span key={key++}>{ws}</span>);
    } else if (punct) {
      parts.push(<span key={key++} className="text-foreground/60">{punct}</span>);
    }
  }
  return parts;
}

// ─── Sub-component: AttemptEntry ──────────────────────────────────────────────

function AttemptEntry({ entry }: { entry: CorrectionEntry }) {
  const formattedSql = formatFailedSQL(entry.sql);

  return (
    <div className="flex flex-col gap-2.5">
      {/* Metadata row — badge IS the attempt number, no redundant label */}
      <div className="flex flex-row items-center gap-2.5">
        <span
          className={cn(
            'inline-flex size-5 items-center justify-center shrink-0',
            'rounded-full bg-muted',
            'font-mono text-[11px] font-medium text-muted-foreground',
          )}
          aria-label={`Attempt ${entry.attempt}`}
        >
          {entry.attempt}
        </span>

        <span className="text-xs text-muted-foreground/30" aria-hidden="true">·</span>

        <span className="font-mono text-xs font-medium text-foreground/70 tracking-tight">
          {entry.error_type}
        </span>
      </div>

      {/* Failed SQL — wraps long lines instead of horizontal scrolling */}
      <div className="w-full">
        <div
          className={cn(
            'card-surface-subtle',
            'py-2 px-3',
          )}
        >
          <pre className="font-mono text-xs whitespace-pre-wrap break-all">
            {highlightFailedSQL(formattedSql)}
          </pre>
        </div>
      </div>

      {/* Error message — quiet commentary on the evidence above */}
      <p className="font-sans text-xs text-muted-foreground/70 leading-relaxed">
        {entry.error_message}
      </p>
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

export function CorrectionPanel({
  corrections,
  defaultOpen = false,
  className,
}: CorrectionPanelProps) {
  const [open, setOpen] = useState(defaultOpen)

  if (!corrections || corrections.length === 0) {
    return null
  }

  return (
    <Collapsible
      open={open}
      onOpenChange={setOpen}
      className={cn('group w-full', className)}
    >
      <CollapsibleTrigger asChild>
        <button
          className={cn(
            'audit-trigger inline-flex items-center gap-2',
            'cursor-pointer select-none',
            'py-2 px-3',
            'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
          )}
          aria-label={`${open ? 'Collapse' : 'Expand'} earlier attempts (${corrections.length})`}
        >
          <motion.span
            animate={{ rotate: open ? 90 : 0 }}
            transition={{ duration: 0.2, ease: 'easeOut' }}
            className="inline-flex items-center"
            aria-hidden="true"
          >
            <ChevronRight className="size-3.5 text-muted-foreground group-hover:text-foreground transition-colors duration-150" />
          </motion.span>

          <span
            className={cn(
              'font-sans text-[13px] font-semibold text-foreground/80',
              'transition-colors duration-150',
              'group-hover:text-foreground',
            )}
          >
            Earlier attempts
          </span>

          <span
            className="font-mono text-xs text-muted-foreground/60 group-hover:text-muted-foreground transition-colors duration-150"
            aria-hidden="true"
          >
            · {corrections.length}
          </span>
        </button>
      </CollapsibleTrigger>

      <CollapsibleContent>
        <div className="mt-2 flex flex-col gap-6">
          {corrections.map((entry) => (
            <AttemptEntry key={entry.attempt} entry={entry} />
          ))}
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}

// ─── v0 preview wrapper ──────────────────────────────────────────────────────

export default function CorrectionPanelPreview() {
  const sampleCorrections: CorrectionEntry[] = [
    {
      attempt: 1,
      sql: `SELECT first_name, last_name FROM Customer WHERE region = 'London';`,
      error_type: 'COLUMN_NOT_FOUND',
      error_message:
        "Column 'first_name' does not exist on table Customer. Did you mean 'name'?",
    },
    {
      attempt: 2,
      sql: `SELECT name, balance FROM Customer WHERE region = 'London';`,
      error_type: 'COLUMN_NOT_FOUND',
      error_message:
        "Column 'balance' does not exist on table Customer. Balance lives on AccountBalance — try joining through Account.",
    },
  ]

  return (
    <div className="min-h-screen bg-background flex items-start justify-center pt-24 px-6">
      <div className="w-full max-w-2xl">
        <CorrectionPanel corrections={sampleCorrections} defaultOpen={true} />
      </div>
    </div>
  )
}