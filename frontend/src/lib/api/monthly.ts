/**
 * Monthly report (`/dashboard/monthly-report/`, AJAX branch). Per-employee
 * monthly attendance rows plus month-level analytics: a daily trend, weekday
 * split, department breakdown, bands and leaderboards. A per-employee calendar
 * drill-down and an Excel export live on sibling endpoints.
 */
import { api, API_BASE } from './client'

export interface MonthlyRow {
  id: number
  name: string
  badge_number: string | null
  department: string
  profile_picture: string | null
  site_id: number | null
  site: string
  category: string
  days_present: number
  days_absent: number
  working_days: number
  late_count: number
  overtime_hours: number
  attendance_percentage: number
  attendance_percentage_raw: number
}

/** A ranked employee in the top/bottom performers leaderboard. */
export interface LeaderEmployee {
  id: number
  name: string
  badge_number: string | null
  department: string
  profile_picture: string | null
  site_id: number | null
  site: string
  category: string
  days_present: number
  working_days: number
  late_count: number
  attendance_percentage: number
}

/** A ranked site in the site performers leaderboard. */
export interface LeaderSite {
  site_id: number
  site: string
  avg_pct: number
  employee_count: number
}

export interface MonthlyReport {
  results: MonthlyRow[]
  summary: {
    total_days: number
    total_employees: number
    avg_attendance: number
    avg_attendance_effective: number
    total_present: number
    total_late: number
    total_overtime: number
    employees_with_zero_attendance: number
    month_name: string
    year: number
    month_num: number
  }
  comparison: {
    previous: { month: string; avg_attendance: number; total_present: number; total_late: number }
    delta_avg_pct: number
    delta_present: number
    delta_late: number
    delta_overtime: number
  }
  trend: { date: string; day: number; weekday: string; present: number; absent: number; late: number }[]
  weekday_split: { weekday: string; avg_present: number; avg_present_pct: number; total_late: number }[]
  department_breakdown: { department: string; employee_count: number; avg_pct: number; total_late: number }[]
  /** Top/bottom employees by effective attendance % (bottom excludes zero-attendance). */
  employee_leaderboard: { top: LeaderEmployee[]; bottom: LeaderEmployee[] }
  /** Top/bottom sites by average attendance %. */
  site_leaderboard: { top: LeaderSite[]; bottom: LeaderSite[] }
  bands: { champion: number; steady: number; at_risk: number; critical: number }
  anomalies: { severity: string; icon: string; title: string; message: string }[]
  pagination: {
    current_page: number
    num_pages: number
    total_items: number
    has_next: boolean
    has_previous: boolean
    start_index: number
    end_index: number
  }
  sites: { id: number; name: string }[]
  employers: string[]
  permissions: { is_superuser: boolean }
}

export interface MonthlyParams {
  month?: number
  year?: number
  site?: string
  employer?: string
  search?: string
  page?: number
  per_page?: number
}

/** Monthly analytics + employee rows. */
export function getMonthlyReport(params: MonthlyParams = {}) {
  return api.get<MonthlyReport>('/dashboard/monthly-report/', { ...params })
}

// ---- Per-employee calendar drill-down ----
export interface MonthlyCalendar {
  employee: { id: number; name: string; badge_number: string | null; department: string; site: string }
  month: number
  year: number
  month_name: string
  num_days: number
  attendance: {
    date: string
    check_in_time: string | null
    check_out_time: string | null
    late_minutes: number
    overtime_hours: number
    status: string
  }[]
}

export function getMonthlyEmployeeCalendar(employeeId: number, params: { month?: number; year?: number }) {
  return api.get<MonthlyCalendar>(`/dashboard/monthly-report/calendar/${employeeId}/`, { ...params })
}

/** Excel export (navigate). */
export function monthlyExportUrl(params: MonthlyParams = {}) {
  const q = new URLSearchParams()
  Object.entries(params).forEach(([k, v]) => v && q.set(k, String(v)))
  return `${API_BASE}/dashboard/monthly-report/export/?${q.toString()}`
}
