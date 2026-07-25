/**
 * Site admins (`/dashboard/site-admins/*`) — superuser-only management of the
 * admin/viewer users and their site assignments.
 *
 * The list has a JSON branch, but the add/edit/delete endpoints are classic
 * form-POST views that **redirect** (no JSON body). We POST form-encoded data
 * and, since the server signals success/failure via a redirect + flash message
 * we can't read cross-origin-style, we treat a completed request as success and
 * refetch the list. (A future backend tweak could add an AJAX JSON branch.)
 */
import { api } from './client'

/** Admin role codes as stored on AdminProfile. */
export type AdminRole = 'site_admin' | 'viewer' | string

export interface SiteAdminRow {
  /** This is the underlying User id. */
  id: number
  username: string
  email: string
  role: AdminRole
  role_display: string
  /** First assigned site id, or '' — legacy compat. */
  site_id: number | ''
  site_ids: number[]
  /** 'All sites' | comma-joined names | 'No Site'. */
  site_name: string
}

export interface SiteAdminsResponse {
  results: SiteAdminRow[]
  sites: { id: number; name: string }[]
  search_query: string
}

/** List admins + the site catalogue for the form. Superuser only. */
export function getSiteAdmins(params: { search?: string } = {}) {
  return api.get<SiteAdminsResponse>('/dashboard/site-admins/', { search: params.search })
}

/** Payload for creating/updating a site admin. */
export interface SiteAdminInput {
  username: string
  email: string
  /** Required on create; omit on edit to keep the current password. */
  password?: string
  role: AdminRole
  /** Site ids (ignored for viewers, who get all sites). */
  sites: number[]
}

function toFormData(input: SiteAdminInput): FormData {
  const fd = new FormData()
  fd.append('username', input.username)
  fd.append('email', input.email)
  if (input.password) fd.append('password', input.password)
  fd.append('role', input.role)
  input.sites.forEach((s) => fd.append('sites', String(s)))
  return fd
}

/** Create a new admin/viewer. Resolves once the server has processed the form. */
export function createSiteAdmin(input: SiteAdminInput) {
  return api.post<unknown>('/dashboard/site-admins/add/', toFormData(input))
}

/** Update an existing admin (by User id). */
export function updateSiteAdmin(id: number, input: SiteAdminInput) {
  return api.post<unknown>(`/dashboard/site-admins/edit/${id}/`, toFormData(input))
}

/** Delete admins in bulk (JSON endpoint). */
export function deleteSiteAdmins(ids: number[]) {
  return api.post<{ message?: string; error?: string }>('/dashboard/site-admins/delete/bulk/', {
    ids,
  })
}
