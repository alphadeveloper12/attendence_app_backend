import { API_BASE } from './api/client'
import { getCsrfToken } from './auth'

/**
 * POST a JSON/body to a file-streaming endpoint and trigger a browser download
 * of the returned file. Used for exports that require a request body (e.g.
 * export-selected), which a plain link can't do.
 */
export async function downloadPost(path: string, body: unknown, fallbackName = 'export.xlsx') {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: {
      'X-Requested-With': 'XMLHttpRequest',
      'X-CSRFToken': getCsrfToken(),
      'Content-Type': 'application/json',
    },
    credentials: 'same-origin',
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`Export failed (${res.status})`)

  // Prefer the server's filename from Content-Disposition.
  const cd = res.headers.get('content-disposition') || ''
  const match = /filename\*?=(?:UTF-8''|")?([^";]+)/i.exec(cd)
  const name = match ? decodeURIComponent(match[1].replace(/"/g, '')) : fallbackName

  const blob = await res.blob()
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = name
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}
