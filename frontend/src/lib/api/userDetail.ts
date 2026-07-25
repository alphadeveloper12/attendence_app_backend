/**
 * Employee detail (`/dashboard/user/<id>/`, AJAX branch) + its sub-resources
 * (status/site history). The main endpoint returns the profile, range stats and
 * either a daily slot view (with geofence pins) or a calendar of days.
 */
import { api } from './client'

/** Employee profile block — the full set the detail page renders. */
export interface EmployeeProfile {
  id: number
  name: string
  email: string | null
  phone: string | null
  badge_number: string | null
  department: string | null
  position: string | null
  site: string | null
  site_id: number | null
  status: string
  profile_picture: string | null
  job_description: string | null
  // Personal
  nationality: string | null
  gender: string | null
  marital_status: string | null
  religion: string | null
  date_of_birth: string | null
  // Employment
  salary_grade?: string | null
  sponsor: string | null
  employer: string | null
  date_of_joining: string | null
  resumption_date: string | null
  last_working_date: string | null
  termination_reason: string | null
  // Passport / identity
  passport_number: string | null
  passport_expiry: string | null
  visa_details: string | null
  visa_expiry_date: string | null
  labor_card_number: string | null
  mol_id: string | null
  // Salary & logistics
  camp: string | null
  transportation: string | null
  gross_salary?: string | null
  // Leave (when status = Leave)
  leave_approval_date: string | null
  leave_start_date: string | null
  leave_end_date: string | null
  leave_type: string | null
}

export interface DetailStats {
  total_records: number
  present: number
  late: number
  absent: number
  sick: number
  total_sick: number
  late_minutes: number
  early_minutes: number
}

export interface DailySlot {
  time_range: string
  status: string | null
  check_in: string | null
  late_minutes?: number
  early_minutes?: number
  latitude: number | null
  longitude: number | null
}

export interface CalendarDayObj {
  date: string
  is_on_leave: boolean
  site_name: string | null
  employee_status: string
  record: { status: string | null; late_minutes: number; early_minutes: number } | null
  slots: { name: string; time: string; status: string }[]
}

export interface UserDetail {
  employee: EmployeeProfile
  stats: DetailStats
  filter: { type: string; start_date: string; end_date: string; current_date: string; today: string }
  // daily branch:
  slots?: Record<string, DailySlot>
  attendance_id?: number | null
  missing_check_in?: boolean
  can_edit_attendance?: boolean
  day_site_id?: number | null
  day_site_name?: string | null
  // range branch:
  calendar_days?: CalendarDayObj[]
}

export interface UserDetailParams {
  filter?: 'daily' | 'weekly' | 'monthly' | 'custom'
  date?: string
  start_date?: string
  end_date?: string
}

/** Employee detail for a date/range. */
export function getUserDetail(userId: number, params: UserDetailParams = {}) {
  return api.get<UserDetail>(`/dashboard/user/${userId}/`, { ...params })
}

// ---- Sub-resources (read) ----
export interface StatusHistory {
  current_status: string
  history: {
    id: number
    old_status: string | null
    new_status: string | null
    leave_start_date: string | null
    leave_end_date: string | null
    resumption_date: string | null
    note: string | null
    changed_at: string | null
    changed_by: string | null
  }[]
}
export function getStatusHistory(id: number) {
  return api.get<StatusHistory>(`/employees/${id}/status-history/`)
}

export interface SiteHistory {
  current_site: string | null
  segments: {
    id: number
    site_name: string | null
    from_date: string | null
    to_date: string | null
    is_current: boolean
    changed_by: string | null
  }[]
}
export function getSiteHistory(id: number) {
  return api.get<SiteHistory>(`/employees/${id}/site-history/`)
}
