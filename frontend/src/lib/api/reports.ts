/**
 * Daily reports (`/api/reports/data/` + `/api/reports/exec-summary/`). One rich
 * endpoint returns the day's attendance rows, KPI stats, comparison baselines,
 * chart series (hourly histogram, site/dept/category breakdowns) and quick
 * lists (late / missing-checkout / geofence). Excel export is a file stream.
 */
import { api, API_BASE } from './client'

export interface ReportStats {
  total: number
  present: number
  absent: number
  late: number
  missing_checkout: number
  geofence_violations: number
  checking_in_now: number
}

export interface ReportRow {
  id: number
  name: string
  badge_number: string | null
  department: string | null
  position: string | null
  profile_picture: string | null
  site: string
  site_id: number | null
  status: 'Present' | 'Absent'
  check_in: string
  check_out: string
  late_minutes: number
  overtime_hours: number
  is_geofence_violation: boolean
  latitude: number | null
  longitude: number | null
}

export interface QuickListItem {
  id: number
  name: string
  site: string
  check_in: string
  badge_number: string | null
  profile_picture: string | null
  department?: string
  late_minutes?: number
  latitude?: number | null
  longitude?: number | null
}

export interface ReportData {
  results: ReportRow[]
  stats: ReportStats
  comparison: {
    yesterday_present: number
    delta_yesterday: number
    same_weekday_avg_present: number
    delta_baseline: number
    avg_checkin_today: string | null
    avg_checkin_baseline: string | null
    avg_checkin_delta_min: number | null
  }
  hourly_histogram: { hour: number; count: number }[]
  site_rows: { name: string; site_id: number; present: number; total: number; late: number; pct: number }[]
  dept_rows: { department: string; present: number; total: number; pct: number }[]
  category_rows: { category: string; present: number; total: number; pct: number }[]
  anomalies: { severity: string; icon: string; title: string; message: string }[]
  late_list: QuickListItem[]
  missing_checkout_list: QuickListItem[]
  geofence_violation_list: QuickListItem[]
  projection: { projected_present: number; method: string } | null
  sites: { id: number; name: string }[]
  positions: string[]
  categories: string[]
  employers: string[]
  selected_date: string
  permissions: { is_superuser: boolean }
  pagination: {
    current_page: number
    num_pages: number
    total_items: number
    has_next: boolean
    has_previous: boolean
    start_index: number
    end_index: number
  }
}

export interface ReportParams {
  date?: string
  site?: string
  status?: string
  position?: string
  department?: string
  category?: string
  employer?: string
  quick?: string
  page?: number
  per_page?: number
}

/** Full daily report payload (charts + stats + rows). */
export function getReportData(params: ReportParams = {}) {
  return api.get<ReportData>('/api/reports/data/', { ...params })
}

export interface ReportBullet {
  icon?: string
  severity?: string
  text: string
}

/** Report exec-summary bullets (AI-polished when enabled, else heuristic). */
export function getReportSummary(params: { date?: string; site?: string; employer?: string } = {}) {
  return api.get<{ narrative: ReportBullet[]; ai_used: boolean; heuristic_fallback: ReportBullet[]; date: string }>(
    '/api/reports/exec-summary/',
    { ...params },
  )
}

/** Excel export of the report (navigate). */
export function reportExportUrl(params: ReportParams = {}) {
  const q = new URLSearchParams()
  Object.entries(params).forEach(([k, v]) => v && q.set(k, String(v)))
  return `${API_BASE}/dashboard/reports/export/?${q.toString()}`
}
