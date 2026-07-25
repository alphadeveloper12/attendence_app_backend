/**
 * Shared API types for the dashboard. Kept close to the JSON the Django
 * endpoints actually return. Domain-specific types (Employee, Report, …) are
 * added to their endpoint modules as those pages are built.
 */

/** A site, as returned by `/me` and most site pickers. */
export interface Site {
  id: number
  name: string
}

/** The 15 nav-visibility keys computed server-side (see context_processors). */
export type NavKey =
  | 'dashboard' | 'reports' | 'monthly_report' | 'distribution_list' | 'departments'
  | 'user_face' | 'attrition_risk' | 'document_expiry' | 'manpower_recs' | 'ask_data'
  | 'geofence_tuning' | 'salary' | 'sites' | 'site_admins' | 'settings'

/** Admin role. Superusers have no AdminProfile. */
export type Role = 'superuser' | 'site_admin' | 'viewer'

/**
 * The current admin's identity, permissions and site scope — the single source
 * of truth for nav gating, read-only mode and site filters. From `/me`.
 */
export interface Me {
  username: string
  is_superuser: boolean
  is_staff: boolean
  role: Role
  is_readonly: boolean
  can_write: boolean
  nav_visible: Record<NavKey, boolean>
  currency_code: string
  /** Sites this admin may see (all for superusers, else assigned). */
  sites: Site[]
}

/** Common server pagination envelope used by the list endpoints. */
export interface Pagination {
  page: number
  per_page: number
  total_pages: number
  total_count: number
  has_next?: boolean
  has_previous?: boolean
}
