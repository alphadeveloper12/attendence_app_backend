/**
 * Global app settings (`/settings/`) + public holidays (`/settings/holidays/`).
 * Superuser-only for writes. A single global singleton — no site scoping.
 */
import { api } from './client'

/** The full settings object (GET) — also the shape returned after a PUT. */
export interface AppSettings {
  // Attendance & shift rules
  default_office_start_time: string
  default_office_end_time: string
  default_worker_start_time: string
  default_worker_end_time: string
  default_office_day_off: string
  default_worker_day_off: string
  late_grace_minutes: number
  half_day_threshold_hours: number
  normal_ot_threshold_minutes: number
  weekend_days: string[]
  // Geofence / location
  default_geofence_radius_meters: number
  gps_accuracy_tolerance_meters: number
  // Master lists
  sponsors: string[]
  employers: string[]
  // Salary
  currency_code: string
  salary_superuser_only: boolean
  gross_formula: Record<string, number>
  // Documents & compliance
  passport_reminder_lead_days: number
  visa_reminder_lead_days: number
  labour_card_reminder_lead_days: number
  mol_reminder_lead_days: number
  expiry_alert_recipients: string[]
  // Data retention
  distribution_snapshot_retention_months: number
  // Navigation feature flags
  nav_visibility: Record<string, boolean>
  updated_at: string
}

/** Fetch the global settings. */
export function getSettings() {
  return api.get<AppSettings>('/settings/')
}

/** Update settings — only the provided keys are applied. Returns the new state. */
export function updateSettings(patch: Partial<AppSettings>) {
  return api.put<{ success: boolean; settings: AppSettings }>('/settings/', patch)
}

// ---- Public holidays ----
export interface Holiday {
  id: number
  name: string
  date: string
  recurring_annually: boolean
}

/** List public holidays (any admin). */
export function getHolidays() {
  return api.get<{ holidays: Holiday[] }>('/settings/holidays/')
}

/** Add a public holiday (superuser). */
export function createHoliday(body: { name: string; date: string; recurring_annually?: boolean }) {
  return api.post<{ success: boolean } & Holiday>('/settings/holidays/', body)
}

/** Delete a public holiday by id (superuser). */
export function deleteHoliday(id: number) {
  return api.delete<{ success: boolean }>(`/settings/holidays/${id}/`)
}
