'use client';

import type { ReactElement } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '@/lib/utils';
import { useTypewriter } from '@/hooks/useTypewriter';

// ─── Types ────────────────────────────────────────────────────────────────────

export interface NarrationProps {
  /** The full prose string to reveal. */
  text: string;
  /** Base ms per character. Default 18. */
  speed?: number;
  /** Ms before typing starts. Default 0. */
  startDelay?: number;
  /** When false, render full text immediately with no animation. Default true. */
  enabled?: boolean;
  /** Called once when the last character has been typed. */
  onComplete?: () => void;
  /** Extra classes applied to the outermost wrapper. */
  className?: string;
}

// ─── Cursor ───────────────────────────────────────────────────────────────────

/**
 * Thin 1.5px × 1.2em vertical bar at the leading edge of the text.
 * Blinks at 1 Hz. Unmounts (not fades) when typing completes.
 */
function TypingCursor(): ReactElement {
  return (
    <motion.span
      aria-hidden="true"
      className={cn(
        'inline-block align-middle ml-0.5',
        'w-[2px] h-[1.1em]',
        'bg-brand',
        'shrink-0',
      )}
      animate={{ opacity: [1, 0, 1] }}
      transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
    />
  );
}

// ─── Narration ────────────────────────────────────────────────────────────────

export function Narration({
  text,
  speed = 18,
  startDelay = 0,
  enabled = true,
  onComplete,
  className,
}: NarrationProps): ReactElement {
  const { displayed, isComplete } = useTypewriter(text, {
    speed,
    startDelay,
    enabled,
    onComplete,
  });

  return (
    <div className={cn(className)}>
      <p
        className={cn(
        'font-sans',
        'text-lg',
        'leading-relaxed',
        'font-normal',
        'text-foreground',
        'tracking-tight',
        'whitespace-pre-wrap',
        'text-left',
      )}
      >
        {displayed}
        <AnimatePresence>
          {!isComplete && <TypingCursor key="cursor" />}
        </AnimatePresence>
      </p>
    </div>
  );
}

// ─── v0 preview wrapper ───────────────────────────────────────────────────────

export default function NarrationPreview(): ReactElement {
  return (
    <div className="dark min-h-screen bg-background flex items-start justify-center pt-24 px-6">
      <Narration
        text="The total balance across all customer accounts is £4.2 million, with 60% concentrated in the top 5 customers. Average account balance sits at £104,000 — a useful benchmark when evaluating individual exposures."
      />
    </div>
  );
}