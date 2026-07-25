import { clsx, type ClassValue } from 'clsx'

/** Tiny className combiner. */
export function cn(...inputs: ClassValue[]) {
  return clsx(inputs)
}

/**
 * Where the Django login page lives. The React home is served by Django, so a
 * plain relative link keeps us on the same origin (mirrors the old
 * `{% url 'admin-login' %}`).
 */
export const LOGIN_URL = '/login/'
export const DEMO_MAILTO =
  'mailto:sales@rocketattendance.com?subject=Book%20a%20demo'
export const PRICING_MAILTO =
  'mailto:sales@rocketattendance.com?subject=Pricing%20enquiry'
