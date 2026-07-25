/**
 * Face enrollment (`/dashboard/user-face/`). A filterable grid of employees
 * showing whether each has a face embedding enrolled. Site-scoped server-side.
 */
import { api } from './client'

export interface FaceEmployee {
  id: number
  name: string
  badge_number: string
  position: string
  site_name: string
  profile_picture: string | null
  face_enrolled: boolean
  detail_url: string
}

export interface FaceEnrollment {
  employees: FaceEmployee[]
  pagination: {
    has_next: boolean
    has_previous: boolean
    current_page: number
    total_pages: number
    total_items: number
    start_index: number
    end_index: number
  }
  filters: { site: string; status: string; emp_status: string; search: string; per_page: number }
  /** Site dropdown options — populated for superusers only, else []. */
  sites: { id: number; name: string }[]
  permissions: { is_superuser: boolean }
  total_count: number
  /** Employment-status breakdown (computed before the emp_status filter). */
  status_counts: Record<string, number>
}

export interface FaceParams {
  site?: string
  /** Face status: 'enrolled' | 'not_enrolled' | 'all'. */
  status?: string
  /** Employment status: 'Active' | 'Leave' | … | 'all'. */
  emp_status?: string
  search?: string
  per_page?: number
  page?: number
}

/** Fetch the face-enrollment grid for the given filters. */
export function getFaceEnrollment(params: FaceParams = {}) {
  return api.get<FaceEnrollment>('/dashboard/user-face/', { ...params })
}
