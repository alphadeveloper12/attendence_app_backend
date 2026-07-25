/**
 * Google Maps helpers. The API key is injected by the Django shell
 * (`react_base.html` → `<meta name="google-maps-key">`) so it never ships in
 * the JS bundle. All maps in the app are read-only (polygon + markers).
 */
import { useJsApiLoader } from '@react-google-maps/api'

/** Read the Maps JS key from the meta tag the server rendered. */
export function getMapsKey(): string {
  if (typeof document === 'undefined') return ''
  return document.querySelector<HTMLMetaElement>('meta[name="google-maps-key"]')?.content ?? ''
}

// A single, stable libraries array — @react-google-maps/api warns if this
// identity changes between renders.
const LIBRARIES: 'geometry'[] = ['geometry']

/**
 * Shared loader for the Google Maps JS SDK. Every map component calls this;
 * the loader dedupes so the script is fetched once. Returns `isLoaded` +
 * `loadError` and whether a key is even configured.
 */
export function useMaps() {
  const key = getMapsKey()
  const { isLoaded, loadError } = useJsApiLoader({
    id: 'google-map-script',
    googleMapsApiKey: key,
    libraries: LIBRARIES,
  })
  return { isLoaded, loadError, hasKey: !!key }
}
