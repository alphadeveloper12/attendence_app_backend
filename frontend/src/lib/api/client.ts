/**
 * Typed HTTP client for the RocketAttendance JSON API.
 *
 * All dashboard data lives under Django's `/api/attendance/*` endpoints and is
 * authenticated with the **Django session cookie** (same-origin). This wrapper:
 *   - prefixes the API base,
 *   - always sends the session cookie (`credentials: 'same-origin'`),
 *   - attaches the CSRF token + `X-Requested-With` (so views take their JSON
 *     branch instead of rendering HTML),
 *   - parses JSON and throws a typed {@link ApiError} on failure,
 *   - redirects to the login page on 401/403 (session expired / not an admin).
 *
 * Endpoint modules (employees.ts, sites.ts, …) build on `api.get/post/…`.
 */
import { getCsrfToken } from '../auth'

/** Base path for every dashboard API call. */
export const API_BASE = '/api/attendance'

/** Where to send the user when their session is missing/expired. */
const LOGIN_PATH = '/login-v2/'

/** Error thrown for any non-2xx API response. Carries status + parsed body. */
export class ApiError extends Error {
  status: number
  body: unknown
  constructor(status: number, message: string, body?: unknown) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.body = body
  }
}

type Query = Record<string, string | number | boolean | null | undefined>

/** Serialize a query object to a `?a=1&b=2` string, skipping empty values. */
function toQuery(query?: Query): string {
  if (!query) return ''
  const parts = Object.entries(query)
    .filter(([, v]) => v !== undefined && v !== null && v !== '')
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`)
  return parts.length ? `?${parts.join('&')}` : ''
}

type RequestOptions = {
  query?: Query
  /** Request body. Plain objects are JSON-encoded; FormData is sent as-is. */
  body?: unknown
  signal?: AbortSignal
}

async function request<T>(
  method: string,
  path: string,
  options: RequestOptions = {},
): Promise<T> {
  const url = `${API_BASE}${path}${toQuery(options.query)}`
  const isForm = options.body instanceof FormData

  const headers: Record<string, string> = {
    'X-Requested-With': 'XMLHttpRequest',
    'X-CSRFToken': getCsrfToken(),
  }
  if (options.body !== undefined && !isForm) headers['Content-Type'] = 'application/json'

  const res = await fetch(url, {
    method,
    headers,
    credentials: 'same-origin',
    signal: options.signal,
    body:
      options.body === undefined
        ? undefined
        : isForm
          ? (options.body as FormData)
          : JSON.stringify(options.body),
  })

  // Session gone or not an admin → bounce to login.
  if (res.status === 401 || res.status === 403) {
    if (typeof window !== 'undefined') window.location.href = LOGIN_PATH
    throw new ApiError(res.status, 'Not authenticated')
  }

  // Parse JSON when present (some endpoints 204 or return files).
  const text = await res.text()
  const data = text ? safeJson(text) : null

  if (!res.ok) {
    const message =
      (data && typeof data === 'object' && 'error' in data && String((data as any).error)) ||
      (data && typeof data === 'object' && 'detail' in data && String((data as any).detail)) ||
      `Request failed (${res.status})`
    throw new ApiError(res.status, message, data)
  }
  return data as T
}

function safeJson(text: string): unknown {
  try {
    return JSON.parse(text)
  } catch {
    return text
  }
}

/** The API surface used across the dashboard. */
export const api = {
  get: <T>(path: string, query?: Query, signal?: AbortSignal) =>
    request<T>('GET', path, { query, signal }),
  post: <T>(path: string, body?: unknown, query?: Query) =>
    request<T>('POST', path, { body, query }),
  put: <T>(path: string, body?: unknown, query?: Query) =>
    request<T>('PUT', path, { body, query }),
  patch: <T>(path: string, body?: unknown, query?: Query) =>
    request<T>('PATCH', path, { body, query }),
  delete: <T>(path: string, query?: Query) => request<T>('DELETE', path, { query }),
}
