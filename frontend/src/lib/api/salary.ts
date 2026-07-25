/**
 * Salary report (`/dashboard/salary-report/`, AJAX branch) — superuser only.
 * Per-employee computed monthly salary (base, OT pays, absence deduction, net).
 * Each row can download a PDF slip from a path-param endpoint.
 */
import { api, API_BASE } from './client'

export interface SalaryRow {
  id: number
  name: string
  badge_number: string
  department: string
  site: string
  profile_picture: string | null
  gross_salary: string
  basic_salary: string
  working_days: number
  present_days: number
  absent_days: number
  normal_ot_hours: number
  special_ot_hours: number
  normal_ot_pay: number
  special_ot_pay: number
  deduction: number
  net_salary: number
}

export interface SalaryReport {
  results: SalaryRow[]
  summary: { month: number; year: number; month_name: string; total_employees: number }
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
}

export interface SalaryParams {
  month?: number
  year?: number
  site?: string
  search?: string
  page?: number
  per_page?: number
}

/** Computed salary rows for the month. Superuser only (server-gated). */
export function getSalaryReport(params: SalaryParams = {}) {
  return api.get<SalaryReport>('/dashboard/salary-report/', { ...params })
}

/** PDF salary-slip URL for one employee/month/year (navigate). */
export function salarySlipUrl(employeeId: number, month: number, year: number) {
  return `${API_BASE}/dashboard/download-salary-slip/${employeeId}/${month}/${year}/`
}
