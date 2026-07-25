/**
 * Departments catalogue (`/api/attendance/admin-departments/`). Superuser-only
 * CRUD used to seed the department dropdowns across the app. Deletes are soft
 * (the department and its positions are hidden, employee history is preserved).
 */
import { api } from './client'

export interface Department {
  id: number
  name: string
  manager_name: string
  is_active: boolean
  /** Active positions filed under this department. */
  positions_count: number
}

export interface DepartmentList {
  count: number
  departments: Department[]
}

/** All active departments, in sheet order. */
export function getDepartments() {
  return api.get<DepartmentList>('/admin-departments/')
}

/** Create (or revive a soft-deleted) department. */
export function createDepartment(body: { name: string; manager_name?: string }) {
  return api.post<{ success: boolean; id: number; name: string; manager_name: string }>(
    '/admin-departments/',
    body,
  )
}

/** Rename / reassign a department's manager. */
export function updateDepartment(id: number, body: { name?: string; manager_name?: string }) {
  return api.put<{ success: boolean; id: number; name: string; manager_name: string }>(
    `/admin-departments/${id}/`,
    body,
  )
}

/** Soft-delete a department (hidden from dropdowns; history kept). */
export function deleteDepartment(id: number) {
  return api.delete<{ success: boolean }>(`/admin-departments/${id}/`)
}
