/**
 * useApi — minimal async-call wrapper with loading, error, and data state.
 *
 * For one-shot calls (init, propose, confirm) where you trigger the call
 * imperatively and care about the result.
 * For ongoing queries with their own UI flow, components manage state directly.
 */

import { useState, useCallback } from "react";

interface UseApiState<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
}

interface UseApiResult<T, Args extends unknown[]> extends UseApiState<T> {
  call: (...args: Args) => Promise<T | null>;
  reset: () => void;
}

export function useApi<T, Args extends unknown[]>(
  fn: (...args: Args) => Promise<T>
): UseApiResult<T, Args> {
  const [state, setState] = useState<UseApiState<T>>({
    data: null,
    error: null,
    loading: false,
  });

  const call = useCallback(
    async (...args: Args): Promise<T | null> => {
      setState({ data: null, error: null, loading: true });
      try {
        const result = await fn(...args);
        setState({ data: result, error: null, loading: false });
        return result;
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Unknown error";
        setState({ data: null, error: message, loading: false });
        return null;
      }
    },
    [fn]
  );

  const reset = useCallback(() => {
    setState({ data: null, error: null, loading: false });
  }, []);

  return { ...state, call, reset };
}