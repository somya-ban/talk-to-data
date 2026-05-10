'use client';

import { useState } from 'react';
import { AnimatePresence, motion, type Variants } from 'framer-motion';
import { Pencil, Trash2 } from 'lucide-react';
import { cn } from '@/lib/utils';

// ─── Types ───────────────────────────────────────────────────────────────────

export interface MetricDefinition {
  /** The metric's identifier (e.g. "total_balance") */
  name: string;
  /** Plain-English description */
  description: string;
  /** The SQL formula that computes this metric */
  sql_formula: string;
}

export interface MetricCardProps {
  /** The metric being displayed */
  metric: MetricDefinition;
  /** Called when the user clicks Edit. Parent should open the editor view. */
  onEdit?: () => void;
  /** Called when the user clicks Delete (or Restore). Receives the new deleted state. */
  onDelete?: (isDeleted: boolean) => void;
  /** Initial deleted state. Default: false */
  isDeleted?: boolean;
  /** Extra classes for the outer wrapper */
  className?: string;
}

// ─── Animation variants ───────────────────────────────────────────────────────

const actionVariants: Variants = {
  hidden: { opacity: 0, scale: 0.92 },
  visible: {
    opacity: 1,
    scale: 1,
    transition: { duration: 0.15, ease: [0.4, 0, 0.2, 1] },
  },
  exit: {
    opacity: 0,
    scale: 0.92,
    transition: { duration: 0.12, ease: [0.4, 0, 0.2, 1] },
  },
};

// ─── Component ────────────────────────────────────────────────────────────────

export function MetricCard({
  metric,
  onEdit,
  onDelete,
  isDeleted: initialDeleted = false,
  className,
}: MetricCardProps) {
  const [deleted, setDeleted] = useState(initialDeleted);

  function handleDelete() {
    setDeleted(true);
    onDelete?.(true);
  }

  function handleRestore() {
    setDeleted(false);
    onDelete?.(false);
  }

  return (
    <article
      className={cn(
        'card-surface relative p-5',
        'transition-all duration-200 ease-out',
        deleted && 'opacity-40',
        className,
      )}
    >
      {/* Top row */}
      <div className="flex items-start justify-between gap-3 mb-3">
        <h2
          className={cn(
            'font-mono text-base font-medium text-foreground leading-snug',
            deleted && 'line-through decoration-foreground/60',
          )}
        >
          {metric.name}
        </h2>

        <div className="flex items-center gap-1 flex-shrink-0 pt-px">
          <AnimatePresence mode="wait" initial={false}>
            {deleted ? (
              <motion.button
                key="restore"
                variants={actionVariants}
                initial="hidden"
                animate="visible"
                exit="exit"
                type="button"
                onClick={handleRestore}
                className="font-mono text-xs text-brand hover:underline focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring rounded-sm px-0.5"
              >
                Restore
              </motion.button>
            ) : (
              <motion.div
                key="actions"
                variants={actionVariants}
                initial="hidden"
                animate="visible"
                exit="exit"
                className="flex items-center gap-1"
              >
                <button
                  type="button"
                  onClick={() => onEdit?.()}
                  aria-label="Edit metric"
                  className={cn(
                    'inline-flex items-center justify-center',
                    'size-7 rounded-full',
                    'text-muted-foreground hover:text-foreground',
                    'hover:bg-muted',
                    'transition-colors duration-150 ease-out',
                    'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
                  )}
                >
                  <Pencil size={14} strokeWidth={1.6} aria-hidden="true" />
                </button>

                <button
                  type="button"
                  onClick={handleDelete}
                  aria-label="Delete metric"
                  className={cn(
                    'inline-flex items-center justify-center',
                    'size-7 rounded-full',
                    'text-muted-foreground hover:text-destructive',
                    'hover:bg-muted',
                    'transition-colors duration-150 ease-out',
                    'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
                  )}
                >
                  <Trash2 size={14} strokeWidth={1.6} aria-hidden="true" />
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Description */}
      <p className="font-sans text-sm text-muted-foreground leading-relaxed mb-3">
        {metric.description}
      </p>

      {/* SQL formula */}
      <div className="card-surface-subtle px-3 py-2">
        <code className="font-mono text-xs text-muted-foreground/80 whitespace-pre-wrap break-words">
          {metric.sql_formula}
        </code>
      </div>
    </article>
  );
}

// ─── v0 preview wrapper ──────────────────────────────────────────────────────

export default function MetricCardPreview() {
  return (
    <div className="min-h-screen bg-background flex items-center justify-center px-6">
      <div className="w-full max-w-md">
        <MetricCard
          metric={{
            name: 'total_balance',
            description:
              'Sum of all account balances across all AccountBalance records, taken as a snapshot of current customer holdings.',
            sql_formula:
              'SELECT SUM(ab.balance) AS total_balance\nFROM AccountBalance ab',
          }}
          onEdit={() => console.log('Edit clicked')}
          onDelete={(d) => console.log('Deleted:', d)}
        />
      </div>
    </div>
  );
}