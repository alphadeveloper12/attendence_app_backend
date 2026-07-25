/**
 * Analytics endpoints (`/api/attendance/analytics/*`). These back the
 * "Analytics" nav group: attrition risk, document expiry, manpower
 * recommendations and ask-the-data. All are deterministic on the server (AI is
 * only an optional polish layer elsewhere).
 */
import { api } from './client'

// ---- Attrition risk ----
export interface AttritionFactor {
  label: string
  points: number
  severity: 'high' | 'medium' | 'low' | string
}
export interface AttritionRow {
  employee_id: number
  name: string
  badge_number: string
  site: string
  department: string
  position: string
  score: number
  band: 'high' | 'medium' | 'low'
  factors: AttritionFactor[]
}
export interface AttritionRisk {
  count: number
  as_of: string
  bands: { high: number; medium: number; low: number }
  rows: AttritionRow[]
}

/** Attrition-risk scores for all employees (optionally scoped/filtered). */
export function getAttritionRisk(params: { site?: string; min_score?: number } = {}) {
  return api.get<AttritionRisk>('/analytics/attrition-risk/', {
    site: params.site,
    min_score: params.min_score,
  })
}

// ---- Document expiry ----
/** One passport/visa document and its urgency bucket for an employee. */
export interface ExpiryDoc {
  type: 'Passport' | 'Visa' | string
  date: string
  /** Days until expiry (negative = already expired). */
  days: number
  bucket: 'expired' | 'critical' | 'urgent' | 'soon' | 'ok'
}
export interface DocumentExpiryRow {
  employee_id: number
  name: string
  badge_number: string
  site: string
  department: string
  docs: ExpiryDoc[]
  worst_bucket: 'expired' | 'critical' | 'urgent' | 'soon'
}
export interface DocumentExpiry {
  count: number
  as_of: string
  buckets: { expired: number; critical: number; urgent: number; soon: number }
  rows: DocumentExpiryRow[]
}

/**
 * Active employees whose passport or visa is expiring (or expired). Scoped by
 * site **id** on the server. `bucket` narrows to one urgency band.
 */
export function getDocumentExpiry(params: { site?: number | string; bucket?: string } = {}) {
  return api.get<DocumentExpiry>('/analytics/document-expiry/', {
    site: params.site,
    bucket: params.bucket,
  })
}

// ---- Manpower recommendations ----
/** A single proposed redeployment from an over-staffed to an under-staffed site. */
export interface ManpowerMove {
  from_site_id: number
  from_site: string
  from_count: number
  to_site_id: number
  to_site: string
  to_count: number
  quantity: number
}
export interface ManpowerRecommendation {
  department: string
  position: string
  total: number
  median: number
  moves: ManpowerMove[]
  /** Total headcount this recommendation would move. */
  impact: number
}
export interface ManpowerRecommendations {
  count: number
  as_of: string
  over_factor: number
  under_factor: number
  recommendations: ManpowerRecommendation[]
}

/**
 * "Sensitivity" is a UI concept mapping to the over/under median factors the
 * endpoint accepts. Stricter = only flag big gaps → fewer moves; aggressive =
 * flag smaller gaps → more moves.
 */
export type ManpowerSensitivity = 'strict' | 'default' | 'aggressive'
export const MANPOWER_PRESETS: Record<ManpowerSensitivity, { over: number; under: number }> = {
  strict: { over: 2.0, under: 0.3 },
  default: { over: 1.5, under: 0.5 },
  aggressive: { over: 1.25, under: 0.6 },
}

/** Rebalancing suggestions comparing each site's headcount to the median. */
export function getManpowerRecommendations(
  params: { site?: number | string; over_factor?: number; under_factor?: number } = {},
) {
  return api.get<ManpowerRecommendations>('/analytics/manpower-recommendations/', {
    site: params.site,
    over_factor: params.over_factor,
    under_factor: params.under_factor,
  })
}

// ---- Ask the data (NL query) ----
/** How the parser (LLM or heuristic) understood the question. */
export interface AskUnderstanding {
  intent: string
  status: string | null
  site: string | null
  position: string | null
  department: string | null
  attendance: string | null
  window: string | null
}
export interface AskResult {
  employee_id: number
  name: string
  badge: string
  department: string
  position: string
  site: string
  status: string
}
export interface AskAnswer {
  question: string
  /** 'llm' when the LLM parsed it, else 'heuristic'. */
  engine: 'llm' | 'heuristic' | string
  as_understood: AskUnderstanding
  count: number
  answer: string
  results: AskResult[]
}

/** Ask a natural-language question about the workforce. */
export function askData(question: string) {
  return api.post<AskAnswer>('/analytics/ask/', { q: question })
}
