import { useState, useEffect, useRef } from 'react';

export interface UseTypewriterOptions {
  speed?: number;
  startDelay?: number;
  enabled?: boolean;
  onComplete?: () => void;
}

export interface UseTypewriterResult {
  displayed: string;
  isComplete: boolean;
}

/**
 * Returns the delay in ms to wait AFTER revealing this character before
 * revealing the next one. Variable cadence — punctuation triggers longer
 * pauses so reading feels natural rather than mechanical.
 */
function delayAfterChar(char: string, baseSpeed: number): number {
  switch (char) {
    case ',':
    case ';':
      return baseSpeed * 4;
    case '.':
    case '?':
    case '!':
      return baseSpeed * 9;
    case ':':
      return baseSpeed * 6;
    case '—':
    case '-':
    case '(':
    case ')':
      return baseSpeed * 3;
    default:
      return baseSpeed;
  }
}

/**
 * Streams `text` character by character with variable cadence on punctuation.
 *
 * Robustness notes:
 * - Survives React 19 StrictMode double-mount: cleanup cancels any pending
 *   timeout and the next run starts fresh from index 0.
 * - Survives `text` prop changes mid-stream: full reset and restart.
 * - Survives `enabled` flipping false: snaps to full text immediately.
 * - `onComplete` fires exactly once via a ref guard, even if React calls
 *   the effect cleanup-then-reinit cycle.
 */
export function useTypewriter(
  text: string,
  options?: UseTypewriterOptions,
): UseTypewriterResult {
  const {
    speed = 18,
    startDelay = 0,
    enabled = true,
    onComplete,
  } = options ?? {};

  const [displayed, setDisplayed] = useState<string>(
    enabled ? '' : text,
  );
  const [isComplete, setIsComplete] = useState<boolean>(!enabled);

  // Keep onComplete in a ref so the effect doesn't restart when it changes.
  const onCompleteRef = useRef(onComplete);
  useEffect(() => {
    onCompleteRef.current = onComplete;
  }, [onComplete]);

  useEffect(() => {
    // Disabled path: snap to the full text and mark complete.
    if (!enabled) {
      setDisplayed(text);
      setIsComplete(true);
      return;
    }

    // Empty text: nothing to type.
    if (text.length === 0) {
      setDisplayed('');
      setIsComplete(true);
      return;
    }

    // Fresh run: reset state.
    setDisplayed('');
    setIsComplete(false);

    let index = 0;
    let cancelled = false;
    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    let completionFired = false;

    function tick() {
      if (cancelled) return;

      index += 1;
      setDisplayed(text.slice(0, index));

      if (index < text.length) {
        // Pause after the character we just revealed.
        const charJustRevealed = text[index - 1];
        const delay = delayAfterChar(charJustRevealed, speed);
        timeoutId = setTimeout(tick, delay);
      } else {
        setIsComplete(true);
        if (!completionFired) {
          completionFired = true;
          onCompleteRef.current?.();
        }
      }
    }

    // Initial delay before the first character appears.
    timeoutId = setTimeout(tick, Math.max(startDelay, speed));

    return () => {
      cancelled = true;
      if (timeoutId !== null) {
        clearTimeout(timeoutId);
        timeoutId = null;
      }
    };
  }, [text, speed, startDelay, enabled]);

  return { displayed, isComplete };
}