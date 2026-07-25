import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { TriangleAlert, MapPin, Download } from 'lucide-react'
import { getAlerts, alertsExportUrl, type AttendanceAlert } from '../../lib/api/dashboard'
import { GlassCard } from '../../components/ui/GlassCard'
import { EmptyState } from '../../components/ui/EmptyState'
import { SkeletonRows } from '../../components/ui/Skeleton'
import { Pill } from '../../components/ui/Pill'
import { Dialog } from '../../components/ui/Dialog'
import { GeofenceMap } from '../../components/ui/GeofenceMap'

/**
 * Alerts panel — today's out-of-bounds check-ins and "marked attendance while
 * on leave" flags. Each alert links to the employee; a map button plots all
 * geolocated alerts on one map. Polls every 60s alongside the dashboard.
 */
export function AlertsPanel({ site }: { site?: string }) {
  const navigate = useNavigate()
  const [mapOpen, setMapOpen] = useState(false)

  const { data, isLoading } = useQuery({
    queryKey: ['alerts', site],
    queryFn: () => getAlerts({ site: site || undefined }),
    refetchInterval: 60_000,
  })

  const alerts = data?.alerts ?? []
  const geoAlerts = useMemo(
    () => alerts.filter((a) => a.lat != null && a.long != null),
    [alerts],
  )

  return (
    <GlassCard className="p-5">
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="grid h-8 w-8 place-items-center rounded-xl bg-black/[0.05] text-fg">
            <TriangleAlert size={16} />
          </span>
          <h2 className="font-semibold text-fg">
            Alerts{alerts.length ? ` · ${alerts.length}` : ''}
          </h2>
        </div>
        <div className="flex items-center gap-1.5">
          {geoAlerts.length > 0 && (
            <button
              type="button"
              onClick={() => setMapOpen(true)}
              className="flex h-8 items-center gap-1.5 rounded-lg border border-black/10 px-3 text-xs font-semibold text-fg hover:bg-black/[0.04]"
            >
              <MapPin size={13} /> Map
            </button>
          )}
          <a
            href={alertsExportUrl({ site: site || undefined })}
            className="grid h-8 w-8 place-items-center rounded-lg border border-black/10 text-fg-muted hover:bg-black/[0.04]"
            title="Export alerts"
          >
            <Download size={14} />
          </a>
        </div>
      </div>

      {isLoading ? (
        <SkeletonRows rows={4} />
      ) : alerts.length === 0 ? (
        <EmptyState title="All clear" message="No geofence or leave alerts today." />
      ) : (
        <ul className="max-h-[380px] space-y-2 overflow-y-auto pr-1">
          {alerts.map((a) => (
            <li key={a.id}>
              <button
                type="button"
                onClick={() => navigate(`/dashboard/user/${a.user_id}-v2`)}
                className="flex w-full items-center gap-3 rounded-xl bg-black/[0.02] p-2.5 text-left transition-colors hover:bg-black/[0.04]"
              >
                <span className="grid h-9 w-9 shrink-0 place-items-center rounded-full bg-black/[0.06] text-xs font-bold text-fg-muted">
                  {a.user_name.slice(0, 2).toUpperCase()}
                </span>
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-semibold text-fg">{a.user_name}</p>
                  <p className="truncate text-xs text-fg-faint">
                    {a.site} · {a.time}
                  </p>
                </div>
                <Pill tone={a.kind === 'geofence' ? 'solid' : 'outline'}>{a.status}</Pill>
              </button>
            </li>
          ))}
        </ul>
      )}

      <Dialog
        open={mapOpen}
        onOpenChange={setMapOpen}
        title="Alert locations"
        description={`${geoAlerts.length} geolocated alert${geoAlerts.length === 1 ? '' : 's'}`}
        className="max-w-2xl"
      >
        <GeofenceMap
          markers={geoAlerts.map((a) => ({ lat: a.lat as number, lng: a.long as number }))}
          height={440}
        />
      </Dialog>
    </GlassCard>
  )
}

/** Small helper: does this alert have coordinates? */
export function hasCoords(a: AttendanceAlert) {
  return a.lat != null && a.long != null
}
