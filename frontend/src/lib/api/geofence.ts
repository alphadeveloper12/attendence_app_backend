/**
 * Geofence tuning (`/analytics/geofence-tuning/`). For each site with a polygon,
 * clusters recent check-in "misses" (points that failed the geofence) so admins
 * can see where the boundary is mis-drawn and by how much.
 */
import { api } from './client'

/** One cluster of missed check-ins near/outside a site boundary. */
export interface GeofenceCluster {
  lat: number
  lon: number
  count: number
  unique_users: number
  avg_distance_m: number
  direction: string | null
  /** 'boundary' | 'near' | 'off_site'. */
  kind: string
  /** 'high' | 'med' | 'low'. */
  severity: string
  suggestion: string | null
}

export interface GeofenceSite {
  site_id: number
  site_name: string
  polygon: [number, number][]
  centroid: [number, number] | null
  total_misses: number
  clusters: GeofenceCluster[]
  has_action_items: boolean
  action_summary: {
    total_misses: number
    near_edge_misses: number
    far_misses: number
    boundary_clusters: number
    off_site_clusters: number
  }
}

export interface GeofenceTuning {
  as_of: string
  window_days: number
  sites: GeofenceSite[]
}

/** Clustered check-in misses per site over the last `days` (1–365). */
export function getGeofenceTuning(params: { days?: number; site?: string } = {}) {
  return api.get<GeofenceTuning>('/analytics/geofence-tuning/', {
    days: params.days,
    site: params.site,
  })
}
