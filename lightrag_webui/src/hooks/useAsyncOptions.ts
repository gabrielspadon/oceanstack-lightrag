import { useCallback, useEffect, useMemo, useState } from 'react'
import { useDebounce } from '@/hooks/useDebounce'

/**
 * How the debounced-search-term auto-fetch effect behaves while `preload`
 * is true. Extracted from AsyncSearch/AsyncSelect, which each gate the
 * auto-fetch differently once preloading:
 * - 'always': no special preload handling — always fetch on debounced
 *   search-term change.
 * - 'skip': never auto-fetch while preloading (AsyncSearch — it populates
 *   preload data via `refetchOnValueChange` and an imperative on-focus
 *   fetch instead).
 * - 'until-loaded': auto-fetch once while preloading and no data has been
 *   loaded yet, then stop (AsyncSelect — a single preload fetch, after
 *   which searching filters the already-fetched options client-side).
 */
export type PreloadAutoFetchPolicy = 'always' | 'skip' | 'until-loaded'

export interface UseAsyncOptionsParams<T> {
  /** Async function to fetch options for a given query string. */
  fetcher: (query?: string) => Promise<T[]>
  /** Current search term driving the debounced fetch/filter. */
  searchTerm: string
  /** Preload all data ahead of time; filters client-side instead of
   *  refetching per keystroke once loaded. */
  preload?: boolean
  /** Function to filter options client-side while in preload mode. */
  filterFn?: (option: T, query: string) => boolean
  /** Debounce delay (ms) applied to non-preload fetches. Preload mode
   *  always debounces at 0, matching both components' original
   *  `useDebounce(preload ? 0 : delay)` call. Default 150. */
  delay?: number
  /** Currently selected value. Only consulted when `refetchOnValueChange`
   *  is set. */
  value?: string | null
  /** Re-fetch keyed on `value` whenever it changes (AsyncSearch's "load
   *  initial value" behavior). Default false — AsyncSelect does not use
   *  this trigger. */
  refetchOnValueChange?: boolean
  /** Preload/auto-fetch interaction policy, see {@link PreloadAutoFetchPolicy}. */
  preloadAutoFetch?: PreloadAutoFetchPolicy
  /** Message used when a thrown fetch error is not an `Error` instance. */
  errorFallbackMessage?: string
}

export interface UseAsyncOptionsResult<T> {
  /** Fetched options, filtered client-side when in preload mode with a
   *  non-empty debounced search term. */
  options: T[]
  /** Raw fetched options before the preload-mode client-side filter. */
  fetchedOptions: T[]
  loading: boolean
  error: string | null
  /** Imperatively (re)fetch for a given query — used by AsyncSearch's
   *  on-focus trigger. */
  fetchOptions: (query: string) => Promise<void>
}

/**
 * Shared async-option state machine extracted from AsyncSearch and
 * AsyncSelect: fetchedOptions/loading/error state, the
 * `useDebounce(preload ? 0 : delay)` debounce, the preload-mode
 * client-side filter, and the fetch error fallback. The two components'
 * differing fetch-trigger policies (refetch on value change, and whether
 * the debounced auto-fetch effect keeps firing once preloaded) are exposed
 * as options rather than baked in, so each component stays a thin
 * consumer of this hook.
 */
export function useAsyncOptions<T>({
  fetcher,
  searchTerm,
  preload,
  filterFn,
  delay = 150,
  value,
  refetchOnValueChange = false,
  preloadAutoFetch = 'always',
  errorFallbackMessage = 'Failed to fetch options'
}: UseAsyncOptionsParams<T>): UseAsyncOptionsResult<T> {
  const [fetchedOptions, setFetchedOptions] = useState<T[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const debouncedSearchTerm = useDebounce(searchTerm, preload ? 0 : delay)

  const fetchOptions = useCallback(
    async (query: string) => {
      try {
        setLoading(true)
        setError(null)
        const data = await fetcher(query)
        setFetchedOptions(data)
      } catch (err) {
        setError(err instanceof Error ? err.message : errorFallbackMessage)
      } finally {
        setLoading(false)
      }
    },
    [fetcher, errorFallbackMessage]
  )

  // Fetch on debounced search-term changes, subject to the preload policy.
  // fetchedOptions.length is read but intentionally excluded from deps (see
  // eslint-disable below) — including it would re-run this effect merely
  // because the fetch it triggers updated that same state.
  useEffect(() => {
    if (preload) {
      if (preloadAutoFetch === 'skip') return
      if (preloadAutoFetch === 'until-loaded' && fetchedOptions.length > 0) return
    }
    // eslint-disable-next-line react-hooks/set-state-in-effect
    fetchOptions(debouncedSearchTerm)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [preload, preloadAutoFetch, debouncedSearchTerm, fetchOptions])

  // Re-fetch keyed on the selected value (AsyncSearch's "load initial
  // value" effect). No-op unless the caller opts in.
  useEffect(() => {
    if (!refetchOnValueChange || !value) return
    // eslint-disable-next-line react-hooks/set-state-in-effect
    fetchOptions(value)
  }, [refetchOnValueChange, value, fetchOptions])

  // In preload mode, derive filtered options without mutating state.
  const options = useMemo(() => {
    if (preload && debouncedSearchTerm) {
      return fetchedOptions.filter((option) => (filterFn ? filterFn(option, debouncedSearchTerm) : true))
    }
    return fetchedOptions
  }, [preload, debouncedSearchTerm, filterFn, fetchedOptions])

  return { options, fetchedOptions, loading, error, fetchOptions }
}
