import { useQuery } from '@tanstack/react-query'
import { useParams, useNavigate } from 'react-router-dom'
import { ArrowLeft, Users, MapPin } from 'lucide-react'
import { getSiteDetail, getSiteCoordinates } from '../../lib/api/sites'
import { PageHeader } from '../../components/ui/PageHeader'
import { GlassCard } from '../../components/ui/GlassCard'
import { StatTile } from '../../components/ui/StatTile'
import { GeofenceMap } from '../../components/ui/GeofenceMap'
import { Skeleton } from '../../components/ui/Skeleton'

/** Strip the `-v2` suffix the routing convention appends to detail URLs. */
function useSiteId(): string {
  const { idv2 = '' } = useParams()
  return idv2.replace(/-v2$/, '')
}

/**
 * Site Detail — a site's header (employee count) and its geofence on a map.
 * The map geometry is fetched separately (polygon + center, with a circular
 * fallback for sites that only have a radius).
 */
export default function SiteDetailPage() {
  const id = useSiteId()
  const navigate = useNavigate()

  const detail = useQuery({ queryKey: ['site', id], queryFn: () => getSiteDetail(id) })
  const coords = useQuery({ queryKey: ['site-coords', id], queryFn: () => getSiteCoordinates(id) })

  const site = detail.data?.site
  const c = coords.data

  return (
    <div>
      <button
        type="button"
        onClick={() => navigate('/dashboard/sites-v2')}
        className="mb-3 inline-flex items-center gap-1.5 text-sm font-medium text-fg-muted transition-colors hover:text-fg"
      >
        <ArrowLeft size={15} /> All sites
      </button>

      <PageHeader
        title={site?.name ?? (detail.isLoading ? 'Loading…' : 'Site')}
        subtitle={c?.has_coordinates ? 'Geofence configured' : 'No geofence drawn'}
      />

      <div className="grid gap-4 lg:grid-cols-[280px_1fr]">
        {/* Facts */}
        <div className="space-y-3">
          {detail.isLoading ? (
            <Skeleton className="h-24 rounded-2xl" />
          ) : (
            <StatTile
              label="Employees"
              value={site?.employee_count?.toLocaleString() ?? '—'}
              icon={<Users size={18} />}
            />
          )}
          <GlassCard className="p-4">
            <div className="flex items-center gap-2 text-sm text-fg-muted">
              <MapPin size={16} className="text-fg-faint" />
              {c?.has_coordinates
                ? `Polygon · ${c.coordinates.length} points`
                : c?.geofence_radius_meters
                  ? `Circular · ${Math.round(c.geofence_radius_meters)} m radius`
                  : 'No location set'}
            </div>
          </GlassCard>
        </div>

        {/* Map */}
        <div>
          {coords.isLoading ? (
            <Skeleton className="h-[420px] rounded-2xl" />
          ) : (
            <GeofenceMap
              polygon={c?.coordinates ?? []}
              center={c?.center ?? null}
              circle={
                c && !c.has_coordinates && c.geofence_lat != null && c.geofence_lng != null && c.geofence_radius_meters
                  ? {
                      center: { lat: c.geofence_lat, lng: c.geofence_lng },
                      radiusMeters: c.geofence_radius_meters,
                    }
                  : null
              }
            />
          )}
        </div>
      </div>
    </div>
  )
}
