/**
 * Sidebar navigation model for the dashboard.
 *
 * Mirrors the legacy `base.html` sidebar: four groups, each item gated by a
 * `nav_visible` key from `/me`. Some items are additionally superuser-only
 * (matching the template's `user.is_superuser` guards). {@link visibleNav}
 * applies the gating for a given `Me`.
 */
import {
  LayoutDashboard, FileBarChart, CalendarRange, Network, Building2, ScanFace,
  TrendingDown, FileClock, Users, MessagesSquare, MapPinned, Wallet, MapPin,
  ShieldCheck, Settings, type LucideIcon,
} from 'lucide-react'
import type { Me, NavKey } from '../lib/api/types'

export interface NavItem {
  key: NavKey
  label: string
  to: string
  icon: LucideIcon
  /** Only superusers ever see this item (in addition to its nav_visible flag). */
  superOnly?: boolean
}

export interface NavGroup {
  label: string
  items: NavItem[]
}

export const NAV_GROUPS: NavGroup[] = [
  {
    label: 'Overview',
    items: [
      { key: 'dashboard', label: 'Dashboard', to: '/dashboard-v2', icon: LayoutDashboard },
      { key: 'reports', label: 'Reports', to: '/dashboard/reports-v2', icon: FileBarChart },
      { key: 'monthly_report', label: 'Monthly Report', to: '/dashboard/monthly-report-v2', icon: CalendarRange },
    ],
  },
  {
    label: 'Workforce',
    items: [
      { key: 'distribution_list', label: 'Distribution', to: '/dashboard/distribution-list-v2', icon: Network },
      { key: 'departments', label: 'Departments', to: '/dashboard/departments-management-v2', icon: Building2 },
      { key: 'user_face', label: 'Face Enrollment', to: '/dashboard/user-face-v2', icon: ScanFace },
    ],
  },
  {
    label: 'Analytics',
    items: [
      { key: 'attrition_risk', label: 'Attrition Risk', to: '/dashboard/attrition-risk-v2', icon: TrendingDown },
      { key: 'document_expiry', label: 'Document Expiry', to: '/dashboard/document-expiry-v2', icon: FileClock },
      { key: 'manpower_recs', label: 'Manpower Recs', to: '/dashboard/manpower-recommendations-v2', icon: Users },
      { key: 'ask_data', label: 'Ask the Data', to: '/dashboard/ask-data-v2', icon: MessagesSquare },
      { key: 'geofence_tuning', label: 'Geofence Tuning', to: '/dashboard/geofence-tuning-v2', icon: MapPinned },
    ],
  },
  {
    label: 'Operations',
    items: [
      { key: 'salary', label: 'Salary', to: '/dashboard/salary-report-v2', icon: Wallet, superOnly: true },
      { key: 'sites', label: 'Sites', to: '/dashboard/sites-v2', icon: MapPin },
      { key: 'site_admins', label: 'Site Admins', to: '/dashboard/site-admins-v2', icon: ShieldCheck, superOnly: true },
      { key: 'settings', label: 'Settings', to: '/dashboard/settings-v2', icon: Settings, superOnly: true },
    ],
  },
]

/** Apply per-user gating: drop items hidden by `nav_visible` or superuser-only. */
export function visibleNav(me: Me): NavGroup[] {
  return NAV_GROUPS.map((group) => ({
    ...group,
    items: group.items.filter(
      (item) =>
        me.nav_visible[item.key] !== false && (!item.superOnly || me.is_superuser),
    ),
  })).filter((group) => group.items.length > 0)
}
