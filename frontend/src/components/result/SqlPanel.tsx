'use client'

import { useState, useEffect, useRef, type ReactNode } from 'react'
import { motion } from 'framer-motion'
import { ChevronRight, Copy, Check } from 'lucide-react'
import {
  Collapsible,
  CollapsibleTrigger,
  CollapsibleContent,
} from '@/components/ui/collapsible'
import { cn } from '@/lib/utils'

// ─── SQL formatter ──────────────────────────────────────────────────────────

/**
 * Pretty-print single-line SQL onto multiple lines along major clause
 * boundaries. If the input is already multi-line (contains \n), pass it
 * through unchanged — backend output that arrives pre-formatted from the
 * LLM is respected as-is.
 *
 * String literals are protected before splitting so a quoted value
 * containing a keyword (e.g. 'SELECT one') can't get mangled.
 */
function formatSQL(sql: string): string {
  if (sql.includes('\n')) return sql

  let s = sql.trim()

  // Protect single-quoted string literals from keyword-splitting regexes.
  const strings: string[] = []
  s = s.replace(/'(?:[^']|'')*'/g, (match) => {
    strings.push(match)
    return `\x00${strings.length - 1}\x00`
  })

  // Top-level clauses — each on its own line. Longest patterns first
  // so "GROUP BY" matches before any standalone keyword could.
  s = s.replace(
    /\s+(GROUP\s+BY|ORDER\s+BY|UNION\s+ALL|FROM|WHERE|HAVING|LIMIT|OFFSET|UNION)\s+/gi,
    (_, kw: string) => '\n' + kw.replace(/\s+/g, ' ').toUpperCase() + ' ',
  )

  // JOIN variants — each on its own line. Longest patterns first so
  // "LEFT OUTER JOIN" wins over "LEFT JOIN" wins over "JOIN".
  s = s.replace(
    /\s+(LEFT\s+OUTER\s+JOIN|RIGHT\s+OUTER\s+JOIN|FULL\s+OUTER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|INNER\s+JOIN|CROSS\s+JOIN|JOIN)\s+/gi,
    (_, kw: string) => '\n' + kw.replace(/\s+/g, ' ').toUpperCase() + ' ',
  )

  // AND / OR — indented continuation under the parent clause.
  s = s.replace(
    /\s+(AND|OR)\s+/gi,
    (_, kw: string) => '\n  ' + kw.toUpperCase() + ' ',
  )

  // Restore protected string literals.
  s = s.replace(/\x00(\d+)\x00/g, (_, idx: string) => strings[parseInt(idx, 10)])

  return s
}

// ─── Syntax highlighting ────────────────────────────────────────────────────

const SQL_KEYWORDS = new Set([
  'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'OUTER',
  'ON', 'AS', 'GROUP', 'BY', 'ORDER', 'HAVING', 'LIMIT', 'OFFSET',
  'AND', 'OR', 'NOT', 'IN', 'LIKE', 'BETWEEN', 'IS', 'NULL', 'CASE',
  'WHEN', 'THEN', 'ELSE', 'END', 'DISTINCT', 'UNION', 'ALL', 'WITH',
  'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'TABLE', 'VIEW', 'INDEX',
  'SUM', 'COUNT', 'AVG', 'MIN', 'MAX', 'DESC', 'ASC',
])

type Token =
  | { kind: 'keyword'; value: string }
  | { kind: 'string'; value: string }
  | { kind: 'number'; value: string }
  | { kind: 'comment'; value: string }
  | { kind: 'other'; value: string }

function tokeniseLine(line: string): Token[] {
  const tokens: Token[] = []

  const trimmed = line.trimStart()
  if (trimmed.startsWith('--')) {
    return [{ kind: 'comment', value: line }]
  }

  const commentIdx = line.indexOf('--')
  const mainPart = commentIdx === -1 ? line : line.slice(0, commentIdx)
  const commentPart = commentIdx === -1 ? null : line.slice(commentIdx)

  const tokenRe = /('(?:[^']|'')*')|(\b\d+(?:\.\d+)?\b)|([A-Za-z_]\w*)|([^\s])|(\s+)/g
  let match: RegExpExecArray | null

  while ((match = tokenRe.exec(mainPart)) !== null) {
    const [, str, num, ident, punct, ws] = match

    if (ws) {
      tokens.push({ kind: 'other', value: ws })
    } else if (str) {
      tokens.push({ kind: 'string', value: str })
    } else if (num) {
      tokens.push({ kind: 'number', value: num })
    } else if (ident) {
      const upper = ident.toUpperCase()
      if (SQL_KEYWORDS.has(upper)) {
        tokens.push({ kind: 'keyword', value: ident })
      } else {
        tokens.push({ kind: 'other', value: ident })
      }
    } else if (punct) {
      tokens.push({ kind: 'other', value: punct })
    }
  }

  if (commentPart !== null) {
    tokens.push({ kind: 'comment', value: commentPart })
  }

  return tokens
}

function highlightSQL(line: string): ReactNode[] {
  const tokens = tokeniseLine(line)

  return tokens.map((tok, i) => {
    switch (tok.kind) {
      case 'keyword':
        return (
          <span key={i} className="text-foreground font-semibold">
            {tok.value}
          </span>
        )
      case 'string':
        return (
          <span key={i} className="text-muted-foreground italic">
            {tok.value}
          </span>
        )
      case 'number':
        return (
          <span key={i} className="text-muted-foreground">
            {tok.value}
          </span>
        )
      case 'comment':
        return (
          <span key={i} className="text-muted-foreground/60">
            {tok.value}
          </span>
        )
      case 'other':
      default:
        return (
          <span key={i} className="text-muted-foreground/80">
            {tok.value}
          </span>
        )
    }
  })
}

// ─── Types ───────────────────────────────────────────────────────────────────

export interface SqlPanelProps {
  /** The SQL query string to display */
  sql: string
  /** Initial open state. Default: false */
  defaultOpen?: boolean
  /** Extra classes for the outermost wrapper */
  className?: string
}

// ─── CopyButton ──────────────────────────────────────────────────────────────

function CopyButton({ sql }: { sql: string }) {
  const [copied, setCopied] = useState(false)
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current)
    }
  }, [])

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(sql)
      setCopied(true)
      timeoutRef.current = setTimeout(() => setCopied(false), 1500)
    } catch {
      // clipboard access denied — fail silently
    }
  }

  return (
    <button
      type="button"
      onClick={handleCopy}
      aria-label={copied ? 'Copied' : 'Copy SQL to clipboard'}
      className={cn(
        'absolute top-2 right-2',
        'size-7 rounded-md',
        'flex items-center justify-center',
        'bg-card hover:bg-muted',
        'border border-border',
        'transition-colors duration-150',
        'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
      )}
    >
      {copied ? (
        <Check className="size-3.5 text-success" />
      ) : (
        <Copy className="size-3.5 text-muted-foreground hover:text-foreground" />
      )}
    </button>
  )
}

// ─── SqlPanel ────────────────────────────────────────────────────────────────

export function SqlPanel({ sql, defaultOpen = false, className }: SqlPanelProps) {
  const [open, setOpen] = useState(defaultOpen)
  const formatted = formatSQL(sql)
  const lines = formatted.split('\n')
  const lineCount = lines.length

  return (
    <Collapsible
      open={open}
      onOpenChange={setOpen}
      className={cn('w-full', className)}
    >
      {/* Trigger row */}
      <CollapsibleTrigger asChild>
        <div
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault()
              setOpen((v) => !v)
            }
          }}
          className={cn(
            'audit-trigger group inline-flex items-center gap-2 py-2 px-3',
            'cursor-pointer select-none',
            'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
          )}
        >
          <motion.span
            animate={{ rotate: open ? 90 : 0 }}
            transition={{ duration: 0.2, ease: 'easeOut' }}
            className="flex items-center shrink-0"
            aria-hidden
          >
            <ChevronRight className="size-3.5 text-muted-foreground group-hover:text-foreground transition-colors duration-150" />
          </motion.span>

          <span
            className={cn(
              'font-mono text-[13px] font-semibold',
              'text-foreground/80 group-hover:text-foreground',
              'transition-colors duration-150',
            )}
          >
            SQL
          </span>

          <span className="font-mono text-xs text-muted-foreground/60 group-hover:text-muted-foreground transition-colors duration-150">
            ·&nbsp;{lineCount}&nbsp;{lineCount === 1 ? 'line' : 'lines'}
          </span>
        </div>
      </CollapsibleTrigger>

      {/* Expandable content */}
      <CollapsibleContent>
        <div className="mt-1">
          <div className="relative card-surface-subtle py-4">
            <CopyButton sql={formatted} />

            <div>
              <div>
                {lines.map((line, idx) => (
                  <div
                    key={idx}
                    className="grid"
                    style={{ gridTemplateColumns: '40px 1fr' }}
                  >
                    <span
                      className={cn(
                        'font-mono text-xs leading-6',
                        'text-muted-foreground/40',
                        'text-right pr-3',
                        'select-none',
                        'shrink-0',
                      )}
                      aria-hidden
                    >
                      {idx + 1}
                    </span>

                    <span
                      className={cn(
                        'font-mono text-xs leading-6',
                        'whitespace-pre-wrap break-all',
                        'pr-10',
                      )}
                    >
                      {highlightSQL(line)}
                      {line === '' && <span>&#8203;</span>}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}

// ─── v0 preview wrapper ──────────────────────────────────────────────────────

export default function SqlPanelPreview() {
  const sampleSql = `SELECT
  c.customer_id,
  c.name,
  SUM(ab.balance) AS total_balance,
  COUNT(DISTINCT a.account_id) AS account_count
FROM Customer c
INNER JOIN Account a ON a.customer_id = c.customer_id
INNER JOIN AccountBalance ab ON ab.account_id = a.account_id
WHERE c.region = 'London'
  AND ab.balance > 0
GROUP BY c.customer_id, c.name
ORDER BY total_balance DESC
LIMIT 10;`

  return (
    <div className="min-h-screen bg-background flex items-start justify-center pt-24 px-6">
      <div className="w-full max-w-2xl">
        <SqlPanel sql={sampleSql} />
      </div>
    </div>
  )
}