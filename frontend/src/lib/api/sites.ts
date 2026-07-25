/**
 * Sites (`/dashboard/sites/*`). The list + detail views have JSON branches
 * (hit via `X-Requested-With`, which the api client always sends). Map geometry
 * for a site comes from a separate coordinates endpoint.
 */
import { api, API_BASE } from './client'
import type { Pagination } from './types'

/** One row in the sites list. */
export interface SiteRow {
  id: number
  name: string
  employee_count: number
  detail_url: string
  office_start: string
  office_end: string
  worker_start: string
  worker_end: string
  office_day_off: string
  worker_day_off: string
  has_geofence: boolean
  geofence_filename: string
}

export interface SitesResponse {
  results: SiteRow[]
  pagination: {
    current_page: number
    num_pages: number
    total_items: number
    has_next: boolean
    has_previous: boolean
    start_index: number
    end_index: number
  }
  permissions: { is_superuser: boolean }
  search_query: string
}

/** Paginated, searchable sites list (with employee counts + geofence flag). */
export function getSites(params: { search?: string; page?: number; per_page?: number } = {}) {
  return api.get<SitesResponse>('/dashboard/sites/', {
    search: params.search,
    page: params.page,
    per_page: params.per_page,
  })
}

/** Minimal site header (id, name, employee_count). */
export function getSiteDetail(id: number | string) {
  return api.get<{ site: { id: number; name: string; employee_count: number } }>(
    `/dashboard/sites/${id}/`,
  )
}

/** Geofence geometry for a site: polygon vertices, center, circular fallback. */
export interface SiteCoordinates {
  success: boolean
  coordinates: { lat: number; lng: number }[]
  center: { lat: number; lng: number }
  has_coordinates: boolean
  geofence_lat: number | null
  geofence_lng: number | null
  geofence_radius_meters: number | null
}

export function getSiteCoordinates(id: number | string) {
  return api.get<SiteCoordinates>(`/api/sites/${id}/coordinates/`)
}

// ---- Site CRUD (superuser) ----
/**
 * Editable site fields. Times are `HH:MM`; day-off is free text (e.g. "Sunday").
 * A geofence polygon is set by uploading a `.kml` file (optional).
 */
export interface SiteFormValues {
  name: string
  office_start: string
  office_end: string
  worker_start: string
  worker_end: string
  office_day_off: string
  worker_day_off: string
  kml_file?: File | null
}

/** Build the multipart body add/edit expect. */
function siteFormData(v: SiteFormValues): FormData {
  const fd = new FormData()
  fd.append('name', v.name)
  fd.append('office_start', v.office_start)
  fd.append('office_end', v.office_end)
  fd.append('worker_start', v.worker_start)
  fd.append('worker_end', v.worker_end)
  fd.append('office_day_off', v.office_day_off)
  fd.append('worker_day_off', v.worker_day_off)
  if (v.kml_file) fd.append('kml_file', v.kml_file)
  return fd
}

/** Create a site. */
export function addSite(v: SiteFormValues) {
  return api.post('/dashboard/sites/add/', siteFormData(v))
}

/** Update a site (partial — only provided fields change). */
export function editSite(id: number, v: SiteFormValues) {
  return api.post(`/dashboard/sites/edit/${id}/`, siteFormData(v))
}

/** Delete one site (resets its employee links). */
export function deleteSite(id: number) {
  return api.post(`/dashboard/sites/delete/${id}/`)
}

/** Delete many sites by id. */
export function bulkDeleteSites(ids: number[]) {
  return api.post<{ message: string }>('/dashboard/sites/delete/bulk/', { ids })
}

// ---- Import ----
/** Bulk-create sites by name from an Excel sheet. */
export function importSites(file: File) {
  const fd = new FormData()
  fd.append('file', file)
  return api.post<{ message: string; imported_count: number; errors: string[] }>('/sites/import/', fd)
}

/** Upsert site schedules (day-off + duty times) from an Excel sheet. */
export function importSiteSchedule(file: File) {
  const fd = new FormData()
  fd.append('file', file)
  return api.post<{ success: boolean; imported_count: number; errors: string[] }>('/sites/import-schedule/', fd)
}

/** Download URL for the schedule Excel template (navigate). */
export function siteScheduleTemplateUrl() {
  return `${API_BASE}/sites/schedule-template/`
}

// `Pagination` is re-exported for pages that want the generic envelope type.
export type { Pagination }
