'use client';

import {
  useRef,
  useEffect,
  useState,
  useCallback,
  type KeyboardEvent,
} from 'react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';
import type { MetricDefinition } from '@/components/metrics/MetricCard';

export type { MetricDefinition } from '@/components/metrics/MetricCard';

// ─── Types ────────────────────────────────────────────────────────────────────

export interface MetricEditorProps {
  /** The metric being edited. Component initialises its state from this. */
  metric: MetricDefinition;
  /** Called when Save is clicked or Enter is pressed in the name field. */
  onSave: (updated: MetricDefinition) => void;
  /** Called when Cancel is clicked or Escape is pressed. */
  onCancel: () => void;
  /** Extra classes for the outer wrapper */
  className?: string;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function estimateRows(text: string, charsPerRow = 72, minRows = 2): number {
  const lineBreaks = (text.match(/\n/g) ?? []).length;
  const wrappedLines = Math.ceil(text.length / charsPerRow);
  return Math.max(minRows, lineBreaks + wrappedLines);
}

// ─── Component ────────────────────────────────────────────────────────────────

export function MetricEditor({
  metric,
  onSave,
  onCancel,
  className,
}: MetricEditorProps) {
  const [name, setName] = useState(metric.name);
  const [description, setDescription] = useState(metric.description);
  const [sqlFormula, setSqlFormula] = useState(metric.sql_formula);

  const nameRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    nameRef.current?.focus();
    const len = nameRef.current?.value.length ?? 0;
    nameRef.current?.setSelectionRange(len, len);
  }, []);

  const handleSave = useCallback(() => {
    onSave({
      name: name.trim() || metric.name,
      description,
      sql_formula: sqlFormula,
    });
  }, [name, description, sqlFormula, metric.name, onSave]);

  const handleFormKeyDown = useCallback(
    (e: KeyboardEvent<HTMLFormElement>) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onCancel();
      }
    },
    [onCancel],
  );

  const handleSubmit = useCallback(
    (e: React.FormEvent<HTMLFormElement>) => {
      e.preventDefault();
      handleSave();
    },
    [handleSave],
  );

  const handleNameKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        handleSave();
      }
    },
    [handleSave],
  );

  const descRows = estimateRows(description, 68, 2);
  const sqlRows = Math.max(3, sqlFormula.split('\n').length);

  return (
    <motion.article
      initial={{ opacity: 0.85, scale: 0.995 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.12, ease: 'easeOut' }}
      className={cn(
        'card-surface relative p-5',
        className,
      )}
    >
      <form onSubmit={handleSubmit} onKeyDown={handleFormKeyDown} noValidate>
        {/* Top row: name input + action cluster */}
        <div className="flex items-start justify-between gap-3 mb-3">
          <input
            ref={nameRef}
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={handleNameKeyDown}
            placeholder={metric.name}
            spellCheck={false}
            autoComplete="off"
            className={cn(
              'flex-1 min-w-0',
              'font-mono text-base font-medium text-foreground leading-snug',
              'bg-transparent border-0 px-2 py-1 -mx-2 -my-1 rounded-md',
              'placeholder:text-muted-foreground/40',
              'transition-colors duration-150 ease-out',
              'hover:bg-muted/40 focus:bg-muted/60',
              'focus:outline-none',
            )}
          />

          <div className="flex items-center gap-2 flex-shrink-0">
            <button
              type="button"
              onClick={onCancel}
              className={cn(
                'font-sans text-sm font-medium',
                'text-muted-foreground hover:text-foreground',
                'bg-transparent',
                'px-3 py-1 rounded-md',
                'transition-colors duration-150 ease-out',
                'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
              )}
            >
              Cancel
            </button>

            <button
              type="submit"
              className={cn(
                'font-sans text-sm font-medium',
                'bg-brand text-brand-foreground',
                'hover:bg-brand/90',
                'px-3 py-1 rounded-md',
                'transition-colors duration-150 ease-out',
                'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-brand focus-visible:ring-offset-2 focus-visible:ring-offset-card',
              )}
            >
              Save
            </button>
          </div>
        </div>

        {/* Description textarea */}
        <textarea
            value={sqlFormula}
            onChange={(e) => setSqlFormula(e.target.value)}
            placeholder="SELECT … AS metric_value&#10;FROM …"
            rows={sqlRows}
            spellCheck={false}
            autoComplete="off"
            autoCorrect="off"
            autoCapitalize="off"
            className={cn(
              'w-full',
              'font-mono text-xs text-muted-foreground/80',
              'bg-transparent border-0 resize-none',
              'whitespace-pre-wrap break-words',
              'p-0',
              'placeholder:text-muted-foreground/40',
              'focus-visible:outline-none',
              'overflow-hidden',
              '[&::-webkit-scrollbar]:hidden',
              '[scrollbar-width:none]',
            )}
            style={{ height: 'auto' }}
            onInput={(e) => {
              const t = e.currentTarget;
              t.style.height = 'auto';
              t.style.height = `${t.scrollHeight}px`;
            }}
          />

        {/* SQL formula textarea */}
        <div
          className={cn(
            'card-surface-subtle px-3 py-2',
            'transition-colors duration-150 ease-out',
          )}
        >
          <textarea
            value={sqlFormula}
            onChange={(e) => setSqlFormula(e.target.value)}
            placeholder="SELECT … AS metric_value&#10;FROM …"
            rows={sqlRows}
            spellCheck={false}
            autoComplete="off"
            autoCorrect="off"
            autoCapitalize="off"
            className={cn(
              'w-full',
              'font-mono text-xs text-muted-foreground/80',
              'bg-transparent border-0 resize-none',
              'whitespace-pre-wrap break-words',
              'p-0',
              'placeholder:text-muted-foreground/40',
              'focus-visible:outline-none',
            )}
          />
        </div>
      </form>
    </motion.article>
  );
}

// ─── v0 preview wrapper ──────────────────────────────────────────────────────

export default function MetricEditorPreview() {
  return (
    <div className="min-h-screen bg-background flex items-center justify-center px-6">
      <div className="w-full max-w-md">
        <MetricEditor
          metric={{
            name: 'total_balance',
            description:
              'Sum of all account balances across all AccountBalance records, taken as a snapshot of current customer holdings.',
            sql_formula:
              'SELECT SUM(ab.balance) AS total_balance\nFROM AccountBalance ab',
          }}
          onSave={(m) => console.log('Save:', m)}
          onCancel={() => console.log('Cancel')}
        />
      </div>
    </div>
  );
}