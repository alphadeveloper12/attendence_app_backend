/**
 * Dashboard data (`/api/attendance/stats`, `/alerts`, `/missing-checkin`,
 * `/analytics/*`). These back the main dashboard's KPIs, filter options,
 * geofence alerts and executive summary. All are site-scoped server-side.
 */
import { api, API_BASE } from './client'

// ---- Stats (KPIs + filter option lists) ----
export interface DashboardStats {
  total_employees: number
  today_attendance_count: number
  late_count: number
  absent_today_count: number
  total_sites: number
  sites: { id: number; name: string }[]
  categories: string[]
  statuses: string[]
  status_counts: Record<string, number>
  employers: string[]
  sponsors: string[]
  chart: { labels: string[]; data: number[] }
}

export interface StatsParams {
  site?: string
  position?: string
  department?: string
  category?: string
  status?: string
  employer?: string
}

/** KPI numbers + filter option lists + a 7-day attendance sparkline. */
export function getStats(params: StatsParams = {}) {
  return api.get<DashboardStats>('/stats/', { ...params })
}

// ---- Geofence / leave alerts ----
export interface AttendanceAlert {
  id: number
  user_name: string
  user_id: number
  user_pic: string | null
  site: string
  site_id: number | null
  badge_number: string
  position: string
  department: string
  time: string
  lat: number | null
  /** NB: server key is `long`, not `lng`. */
  long: number | null
  status: string
  kind: 'geofence' | 'on_leave' | string
}

/** Today's (or `date`'s) out-of-bounds + on-leave attendance alerts. */
export function getAlerts(params: { date?: string; site?: string } = {}) {
  return api.get<{ date: string; alerts: AttendanceAlert[] }>('/alerts/', { ...params })
}

// ---- Missing check-ins ----
export interface MissingCheckin {
  id: number
  employee: string
  employee_id: number
  badge_number: string
  site: string
  date: string
  check_out: string
}

export function getMissingCheckins(params: { date?: string } = {}) {
  return api.get<{ count: number; rows: MissingCheckin[] }>('/missing-checkin/', { ...params })
}

// ---- Executive summary (5 NL bullets) ----
export interface ExecBullet {
  icon: string
  severity: 'good' | 'neutral' | 'warn' | 'bad' | string
  metric: number
  delta: number | null
  text: string
}
export interface ExecSummary {
  as_of: string
  period: string
  engine: 'heuristic' | 'llm' | string
  bullets: ExecBullet[]
}

export function getExecutiveSummary(params: { period?: 'daily' | 'weekly'; ai?: boolean } = {}) {
  return api.get<ExecSummary>('/analytics/executive-summary/', {
    period: params.period,
    ai: params.ai ? '1' : undefined,
  })
}

// ---- Site activity ----
export interface SiteActivityRow {
  site_id: number
  site_name: string
  active_employees: number
  present_today: number
  present_rate: number
  last_attendance: string | null
  hours_since_last: number | null
  activity_status: 'active' | 'idle' | 'inactive' | 'never_used' | string
  trend_7d: number[]
  avg_7d: number
  delta_vs_avg: number
}
export interface SiteActivity {
  as_of: string
  totals: Record<string, number>
  sites: SiteActivityRow[]
  inactive_sites: { site_id: number; site_name: string; activity_status: string }[]
}

export function getSiteActivity(params: { days?: number } = {}) {
  return api.get<SiteActivity>('/analytics/site-activity/', { ...params })
}

/** Alerts export URL (Excel; navigate). */
export function alertsExportUrl(params: { date?: string; site?: string } = {}) {
  const q = new URLSearchParams()
  Object.entries(params).forEach(([k, v]) => v && q.set(k, String(v)))
  return `${API_BASE}/alerts/export/?${q.toString()}`
}
