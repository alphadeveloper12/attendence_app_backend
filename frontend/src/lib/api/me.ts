import { api } from './client'
import type { Me } from './types'

/** Fetch the current admin's identity + permissions + site scope (`/me`). */
export function getMe(signal?: AbortSignal): Promise<Me> {
  return api.get<Me>('/me/', undefined, signal)
}
