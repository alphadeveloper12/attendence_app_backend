import { useQuery } from '@tanstack/react-query'
import { getMe } from '../api/me'
import type { Me } from '../api/types'

/**
 * Load the current admin (`/me`). Cached for the session and reused by the
 * layout (nav gating), route guard (auth), and pages (site scope, read-only).
 * `retry: false` so an unauthenticated 401 fails fast and the guard redirects.
 */
export function useMe() {
  return useQuery<Me>({
    queryKey: ['me'],
    queryFn: ({ signal }) => getMe(signal),
    staleTime: 5 * 60 * 1000,
    retry: false,
  })
}
