import { useCallback, useMemo } from 'react'
import { GoogleMap, Polygon, Marker, Circle } from '@react-google-maps/api'
import { useMaps } from '../../lib/maps'
import { MapPin } from 'lucide-react'

/** A lat/lng point. */
export interface LatLng {
  lat: number
  lng: number
}

/** A cluster of check-in misses, sized by `count`, coloured by `severity`. */
export interface MapCluster extends LatLng {
  count: number
  severity?: 'high' | 'med' | 'low' | string
  label?: string
}

/** Severity → marker fill (monochrome: darker = worse). */
const severityFill: Record<string, string> = {
  high: '#14141a',
  med: '#52525b',
  low: '#a1a1aa',
}

/**
 * Read-only geofence map: draws a site's polygon (or a circular geofence) plus
 * optional point markers or count-scaled clusters. Auto-fits the viewport to
 * everything shown. Used by Site Detail and Geofence Tuning.
 *
 * Renders graceful fallbacks when no key is configured, the SDK fails, or there
 * is simply no geometry to show — so callers never see a blank grey box.
 */
export function GeofenceMap({
  polygon = [],
  center,
  markers = [],
  clusters = [],
  circle,
  height = 420,
}: {
  polygon?: LatLng[]
  center?: LatLng | null
  markers?: LatLng[]
  clusters?: MapCluster[]
  /** Circular geofence fallback when there is no polygon. */
  circle?: { center: LatLng; radiusMeters: number } | null
  height?: number
}) {
  const { isLoaded, loadError, hasKey } = useMaps()

  const resolvedCenter = useMemo<LatLng>(() => {
    if (center) return center
    if (polygon.length) {
      const lat = polygon.reduce((s, p) => s + p.lat, 0) / polygon.length
      const lng = polygon.reduce((s, p) => s + p.lng, 0) / polygon.length
      return { lat, lng }
    }
    if (circle) return circle.center
    return { lat: 25.2048, lng: 55.2708 } // Dubai fallback
  }, [center, polygon, circle])

  // Fit the map to all geometry once it mounts.
  const onLoad = useCallback(
    (map: google.maps.Map) => {
      const pts = [...polygon, ...markers, ...clusters]
      if (circle) pts.push(circle.center)
      if (pts.length < 2) {
        map.setCenter(resolvedCenter)
        map.setZoom(circle ? 15 : 16)
        return
      }
      const bounds = new google.maps.LatLngBounds()
      pts.forEach((p) => bounds.extend(p))
      map.fitBounds(bounds, 48)
    },
    [polygon, markers, clusters, circle, resolvedCenter],
  )

  const hasGeometry = polygon.length > 0 || markers.length > 0 || clusters.length > 0 || !!circle

  if (!hasKey) return <MapFallback height={height} text="Map unavailable — no API key configured." />
  if (loadError) return <MapFallback height={height} text="Map failed to load." />
  if (!isLoaded) return <MapFallback height={height} text="Loading map…" pulse />
  if (!hasGeometry) return <MapFallback height={height} text="No location set for this site." />

  return (
    <div className="overflow-hidden rounded-2xl border border-black/10" style={{ height }}>
      <GoogleMap
        mapContainerStyle={{ width: '100%', height: '100%' }}
        center={resolvedCenter}
        zoom={16}
        onLoad={onLoad}
        options={{
          mapTypeControl: false,
          streetViewControl: false,
          fullscreenControl: false,
          gestureHandling: 'cooperative',
          styles: MAP_STYLE,
        }}
      >
        {polygon.length > 2 && (
          <Polygon
            paths={polygon}
            options={{
              fillColor: '#14141a',
              fillOpacity: 0.08,
              strokeColor: '#14141a',
              strokeOpacity: 0.8,
              strokeWeight: 2,
            }}
          />
        )}

        {circle && !polygon.length && (
          <Circle
            center={circle.center}
            radius={circle.radiusMeters}
            options={{
              fillColor: '#14141a',
              fillOpacity: 0.08,
              strokeColor: '#14141a',
              strokeOpacity: 0.8,
              strokeWeight: 2,
            }}
          />
        )}

        {markers.map((m, i) => (
          <Marker key={`m${i}`} position={m} />
        ))}

        {clusters.map((c, i) => (
          <Marker
            key={`c${i}`}
            position={c}
            label={{ text: String(c.count), color: '#fff', fontSize: '11px', fontWeight: '700' }}
            icon={{
              path: google.maps.SymbolPath.CIRCLE,
              // Scale radius by count (clamped) so hotspots read at a glance.
              scale: Math.min(22, 9 + Math.log2(c.count + 1) * 3),
              fillColor: severityFill[c.severity ?? 'low'] ?? severityFill.low,
              fillOpacity: 0.9,
              strokeColor: '#fff',
              strokeWeight: 1.5,
            }}
          />
        ))}
      </GoogleMap>
    </div>
  )
}

/** Placeholder shown in place of the map (loading / no key / no geometry). */
function MapFallback({ height, text, pulse }: { height: number; text: string; pulse?: boolean }) {
  return (
    <div
      className="grid place-items-center rounded-2xl border border-black/10 bg-black/[0.03] text-center"
      style={{ height }}
    >
      <div className={pulse ? 'animate-pulse' : ''}>
        <MapPin className="mx-auto mb-2 text-fg-faint" size={22} />
        <p className="text-sm text-fg-muted">{text}</p>
      </div>
    </div>
  )
}

// Muted, monochrome-leaning map style to match the light-glass theme.
const MAP_STYLE: google.maps.MapTypeStyle[] = [
  { featureType: 'poi', elementType: 'labels', stylers: [{ visibility: 'off' }] },
  { featureType: 'transit', stylers: [{ visibility: 'off' }] },
  { featureType: 'road', elementType: 'labels.icon', stylers: [{ visibility: 'off' }] },
]
