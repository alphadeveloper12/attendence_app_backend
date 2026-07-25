/** Read Django's CSRF token — from the meta tag first, then the cookie. */
export function getCsrfToken(): string {
  const meta = document.querySelector<HTMLMetaElement>('meta[name="csrf-token"]')
  if (meta?.content) return meta.content
  const m = document.cookie.match(/(?:^|;\s*)csrftoken=([^;]+)/)
  return m ? decodeURIComponent(m[1]) : ''
}

export type LoginResult =
  | { ok: true; redirectUrl: string }
  | { ok: false; error: string }

/**
 * Session login against the existing Django view (`POST /login/`).
 * Mirrors the legacy AJAX contract: sends X-Requested-With + CSRF, receives
 * `{ success, redirect_url }` or `{ success: false, error }`. On success the
 * Django session cookie is set and we can navigate to the dashboard.
 */
export async function sessionLogin(
  username: string,
  password: string,
): Promise<LoginResult> {
  const body = new FormData()
  body.append('username', username)
  body.append('password', password)
  body.append('csrfmiddlewaretoken', getCsrfToken())

  try {
    const res = await fetch('/login/', {
      method: 'POST',
      headers: {
        'X-Requested-With': 'XMLHttpRequest',
        'X-CSRFToken': getCsrfToken(),
      },
      credentials: 'same-origin',
      body,
    })
    const data = await res.json().catch(() => ({}))
    if (data?.success) return { ok: true, redirectUrl: data.redirect_url || '/dashboard/' }
    return { ok: false, error: data?.error || 'Invalid credentials or not an admin user' }
  } catch {
    return { ok: false, error: 'Connection failure. Please try again.' }
  }
}
