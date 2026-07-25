/**
 * Employees (`/api/attendance/employees/*`) — the workforce table plus the
 * add/edit form and bulk actions that back the main dashboard. The list is
 * DRF-paginated and site-scoped server-side.
 */
import { api, API_BASE } from './client'

/** A row in the employee table (EmployeeListLightSerializer). */
export interface EmployeeRow {
  id: number
  name: string
  email: string | null
  phone: string | null
  department: string | null
  position: string | null
  profile_picture_url: string | null
  site: number | null
  site_name: string | null
  salary_grade: string | null
  badge_number: string | null
  mol_id: string | null
  labor_card_number: string | null
  sponsor: string | null
  employer: string | null
  nationality: string | null
  status: string | null
  category: string
  gross_salary: string | null
  basic_salary: string | null
  passport_expiry: string | null
  date_of_joining: string | null
}

/** DRF page envelope. */
export interface Paginated<T> {
  count: number
  next: string | null
  previous: string | null
  results: T[]
}

export interface EmployeeListParams {
  site?: string
  status?: string
  department?: string
  position?: string
  category?: string
  employer?: string
  search?: string
  /** today's attendance: 'present' | 'late' | 'absent'. */
  attendance_filter?: string
  per_page?: number
  page?: number
}

/** Paginated, filtered, site-scoped employee list. */
export function getEmployees(params: EmployeeListParams = {}) {
  return api.get<Paginated<EmployeeRow>>('/employees/', { ...params })
}

/** Hard-delete employees by id (superuser). */
export function bulkDeleteEmployees(ids: number[]) {
  return api.post<{ success: boolean; message: string }>('/employees/delete/bulk/', { ids })
}

/** Create an employee. `body` is the flat field map (see EmployeeFormValues). */
export function addEmployee(body: Record<string, unknown>) {
  return api.post<{ success: boolean; message: string }>('/employees/add/', body)
}

/** Fetch an employee's fields pre-stringified for the edit form. */
export function getEmployeeForEdit(id: number) {
  return api.get<Record<string, string>>(`/employees/edit/${id}/`)
}

/** Update an employee (PUT). Only sent keys are applied for guarded fields. */
export function updateEmployee(id: number, body: Record<string, unknown>) {
  return api.put<{ success: boolean; message: string }>(`/employees/edit/${id}/`, body)
}

// ---- Import (Excel) ----
/** One detected column in the import preview. */
export interface ImportColumn {
  column_index: number
  header: string
  matched_field: string | null
  confidence: number
  status: 'strong' | 'likely' | 'weak' | 'unknown' | string
}
export interface ImportPreview {
  header_row: number
  columns: ImportColumn[]
  available_fields: string[]
}

/** Auto-detect the column mapping for an Excel file (no DB writes). */
export function importPreview(file: File) {
  const fd = new FormData()
  fd.append('file', file)
  return api.post<ImportPreview>('/employees/import-preview/', fd)
}

/** Run the import (the server auto-maps columns and upserts). */
export function importEmployees(file: File) {
  const fd = new FormData()
  fd.append('file', file)
  return api.post<Record<string, unknown>>('/employees/import/', fd)
}

// ---- Bulk edit (Excel template) ----
/**
 * Result of applying a bulk-edit sheet. Rows are keyed on the `Badge ID`
 * column; empty cells leave a value unchanged. Rows without a badge are
 * skipped; unmatched/failed rows are reported in `errors`.
 */
export interface BulkEditResult {
  success: boolean
  total_rows: number
  updated_count: number
  skipped_count: number
  errors: { row: number; badge_number?: string; error: string }[]
}

/** Download URL for the pre-filled bulk-edit Excel template (navigate). */
export function bulkEditTemplateUrl() {
  return `${API_BASE}/employees/bulk-edit/template/`
}

/** Apply an Excel bulk-edit sheet. Salary columns require superuser server-side. */
export function bulkEditEmployees(file: File) {
  const fd = new FormData()
  fd.append('file', file)
  return api.post<BulkEditResult>('/employees/bulk-edit/', fd)
}

// ---- Export URLs (file streams — navigate, don't fetch) ----
/** All employees matching the current filters → Excel. */
export function exportEmployeesUrl(params: EmployeeListParams = {}) {
  const q = new URLSearchParams()
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== '' && v !== null) q.set(k, String(v))
  })
  return `${API_BASE}/employees/export-filtered/?${q.toString()}`
}

/** Enum constants surfaced in the UI (from the Django model choices). */
export const EMPLOYER_CHOICES = ['PIC', 'KFD', 'Kami', 'PRMC'] as const
export const SPONSOR_CHOICES = [
  'Parkway', 'Katilink', 'ReadyMix', 'Mayadan', 'Jafza', 'Golden', 'Old Emp',
] as const
export const MASTER_STATUSES = [
  'Active', 'Leave', 'Resigned', 'Terminated', 'No Renewal', 'Absconding', 'Other',
] as const
export const CATEGORIES = ['staff', 'worker'] as const
export const TRANSPORT_OPTIONS = ['Company Bus', 'personal'] as const
export const LEAVE_TYPES = ['Annual', 'Emergency', 'Unpaid', 'Hajj-Umrah'] as const
