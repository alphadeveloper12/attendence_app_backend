/**
 * Manpower distribution (`/api/attendance/distribution/`). A dept-wise /
 * trade-wise / project-wise headcount matrix, in three flavours:
 *   - `staff`    — site × position matrix (staff category)
 *   - `worker`   — site × position matrix (worker category)
 *   - `resource` — flat per-position totals, no site breakdown
 *
 * The server returns pre-shaped rows (dept header / trade / subtotal) so the
 * page just paints them; the Excel export is a separate server-side download.
 */
import { API_BASE } from './client'
import { api } from './client'

export type DistributionType = 'staff' | 'worker' | 'resource'

/** A row in the distribution table. `kind` decides how it renders. */
export interface DistributionRow {
  kind: 'dept' | 'trade' | 'subtotal'
  /** dept rows: the department name. */
  name?: string
  /** trade/subtotal rows carry their owning department. */
  department?: string
  /** staff/worker: per-site counts keyed by site id (as a string). */
  counts?: Record<string, number>
  /** staff/worker: employees on leave (not attributed to a site). */
  leave?: number
  /** staff/worker: row total (site counts + leave). */
  total?: number
  /** resource: flat count for the row. */
  count?: number
}

export interface Distribution {
  employee_type: DistributionType
  /** Site columns (empty for the resource tab). */
  sites: { id: number; name: string }[]
  rows: DistributionRow[]
  grand_total: number
  selected_site?: string
  /** True when viewing a frozen past snapshot (a date before today was chosen). */
  historical?: boolean
  /** The snapshot's actual date (may predate the requested date if none exact). */
  snapshot_date?: string
  /** The date the user asked for (historical mode). */
  requested_date?: string
  /** Server-supplied banner text for the historical/empty case. */
  message?: string
}

/** Fetch the distribution matrix for one tab, optionally scoped to a site. */
export function getDistribution(params: {
  type: DistributionType
  site?: number | string
  date?: string
}) {
  return api.get<Distribution>('/distribution/', {
    type: params.type,
    site: params.site,
    date: params.date,
  })
}

/**
 * Absolute URL for the server-side Excel export of the current view. Navigate
 * to it (it streams a file with the session cookie) rather than fetching.
 */
export function distributionExportUrl(params: {
  type: DistributionType
  site?: number | string
  date?: string
}) {
  const q = new URLSearchParams({ type: params.type })
  if (params.site) q.set('site', String(params.site))
  if (params.date) q.set('date', params.date)
  return `${API_BASE}/distribution/export/?${q.toString()}`
}
