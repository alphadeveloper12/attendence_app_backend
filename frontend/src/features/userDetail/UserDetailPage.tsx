import { useState, type ReactNode } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useParams, useNavigate } from 'react-router-dom'
import { ArrowLeft, Pencil, UserCheck, Clock, UserX, Stethoscope, Download } from 'lucide-react'
import { getUserDetail, getStatusHistory, getSiteHistory, type UserDetailParams } from '../../lib/api/userDetail'
import { getSiteCoordinates } from '../../lib/api/sites'
import { useMe } from '../../lib/hooks/useMe'
import { PageHeader } from '../../components/ui/PageHeader'
import { GlassCard } from '../../components/ui/GlassCard'
import { StatTile } from '../../components/ui/StatTile'
import { Skeleton } from '../../components/ui/Skeleton'
import { Pill } from '../../components/ui/Pill'
import { Select, DatePicker } from '../../components/ui/Field'
import { Tabs } from '../../components/ui/Tabs'
import { GeofenceMap } from '../../components/ui/GeofenceMap'
import { AttendanceCalendar, CalendarLegend } from '../../components/ui/AttendanceCalendar'
import { API_BASE } from '../../lib/api/client'

const FILTERS = [
  { value: 'daily', label: 'Daily' },
  { value: 'weekly', label: 'Weekly' },
  { value: 'monthly', label: 'Monthly' },
]
const todayISO = () => new Date().toISOString().slice(0, 10)

/** Strip the `-v2` suffix from the route param → numeric id. */
function useEmployeeId(): number {
  const { idv2 = '' } = useParams()
  return Number(idv2.replace(/-v2$/, ''))
}

/**
 * Employee Detail — profile, range attendance stats, and an attendance view
 * that switches between a daily slot breakdown (with a geofence map of the
 * check-in pins) and a calendar for weekly/monthly ranges. Superusers can edit
 * the employee and view status/site history.
 */
export default function UserDetailPage() {
  const id = useEmployeeId()
  const navigate = useNavigate()
  const { data: me } = useMe()
  const [filter, setFilter] = useState<'daily' | 'weekly' | 'monthly'>('daily')
  const [date, setDate] = useState(todayISO())

  const params: UserDetailParams = { filter, date }
  const { data, isLoading } = useQuery({
    queryKey: ['user-detail', id, filter, date],
    queryFn: () => getUserDetail(id, params),
  })

  const emp = data?.employee
  const stats = data?.stats

  return (
    <div>
      <button type="button" onClick={() => navigate(-1)}
        className="mb-3 inline-flex items-center gap-1.5 text-sm font-medium text-fg-muted transition-colors hover:text-fg">
        <ArrowLeft size={15} /> Back
      </button>

      <PageHeader
        title={emp?.name ?? (isLoading ? 'Loading…' : 'Employee')}
        subtitle={emp ? `#${emp.badge_number || '—'} · ${emp.position || '—'} · ${emp.site || '—'}` : undefined}
        actions={
          <div className="flex items-center gap-2">
            <a href={`${API_BASE}/employees/export/?user_id=${id}`}
              className="flex h-10 items-center gap-1.5 rounded-xl border border-black/10 bg-white/60 px-3.5 text-sm font-semibold text-fg hover:bg-white">
              <Download size={15} /> Attendance
            </a>
            {me?.can_write && (
              <button type="button" onClick={() => navigate(`/dashboard/employee-edit/${id}-v2`)}
                className="flex h-10 items-center gap-1.5 rounded-xl bg-[#14141a] px-4 text-sm font-semibold text-white hover:opacity-90">
                <Pencil size={15} /> Edit
              </button>
            )}
          </div>
        }
      />

      {/* Stats */}
      <div className="mb-5 grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatTile label="Present" value={stats?.present ?? 0} icon={<UserCheck size={18} />} />
        <StatTile label="Late" value={stats?.late ?? 0} icon={<Clock size={18} />} />
        <StatTile label="Absent" value={stats?.absent ?? 0} icon={<UserX size={18} />} />
        <StatTile label="Sick (all-time)" value={stats?.total_sick ?? 0} icon={<Stethoscope size={18} />} />
      </div>

      <div className="grid gap-4 lg:grid-cols-[320px_1fr]">
        {/* Profile */}
        <GlassCard className="h-fit p-5">
          <div className="mb-4 flex items-center gap-3">
            {emp?.profile_picture ? (
              <img src={emp.profile_picture} alt="" className="h-14 w-14 rounded-full object-cover" />
            ) : (
              <span className="grid h-14 w-14 place-items-center rounded-full bg-black/[0.06] text-lg font-bold text-fg-muted">
                {(emp?.name ?? '').slice(0, 2).toUpperCase()}
              </span>
            )}
            <div className="min-w-0">
              <p className="truncate font-semibold text-fg">{emp?.name}</p>
              {emp && <Pill tone={emp.status === 'Active' ? 'solid' : 'soft'}>{emp.status}</Pill>}
            </div>
          </div>
          {/* Contact (shown in the hero of the legacy page, not the details grid) */}
          {emp && (emp.email || emp.phone) && (
            <div className="mb-4 space-y-1 border-b border-black/[0.05] pb-3 text-sm">
              {emp.email && <p className="truncate text-fg-muted">{emp.email}</p>}
              {emp.phone && <p className="text-fg-muted">{emp.phone}</p>}
            </div>
          )}
          {isLoading ? (
            <Skeleton className="h-48 rounded-xl" />
          ) : emp ? (
            <div className="space-y-4 text-sm">
              {/* Same five categories as the legacy user-detail page. */}
              <Section title="Personal Details">
                <Row label="Nationality" value={emp.nationality} />
                <Row label="Gender" value={emp.gender} />
                <Row label="Marital Status" value={emp.marital_status} />
                <Row label="Religion" value={emp.religion} />
                <Row label="Date of Birth" value={emp.date_of_birth} />
              </Section>

              <Section title="Employment Info">
                <Row label="Division" value={emp.department} />
                <Row label="Present Designation" value={emp.position} />
                {me?.is_superuser && emp.salary_grade && <Row label="Category" value={emp.salary_grade} />}
                <Row label="Sponsor" value={emp.sponsor} />
                <Row label="Employer" value={emp.employer} />
                <Row label="D.O.J." value={emp.date_of_joining} />
                {emp.resumption_date && <Row label="Last Resumption Date" value={emp.resumption_date} />}
                {emp.last_working_date && <Row label="Last Working Date" value={emp.last_working_date} />}
                {emp.termination_reason && <Row label="Reason" value={emp.termination_reason} />}
                {emp.status === 'Leave' && (
                  <>
                    <Row label="Leave Approval Date" value={emp.leave_approval_date} />
                    <Row label="Leave Start Date" value={emp.leave_start_date} />
                    <Row label="Leave End Date" value={emp.leave_end_date} />
                    <Row label="Leave Type" value={emp.leave_type} />
                  </>
                )}
              </Section>

              <Section title="Passport & Identity">
                <Row label="Passport Nr." value={emp.passport_number} />
                <Row label="Passport Expiry" value={emp.passport_expiry} />
                <Row label="Visa Details" value={emp.visa_details} />
                <Row label="Visa Expiry" value={emp.visa_expiry_date} />
              </Section>

              <Section title="Labour Card Details">
                <Row label="L.Card / CEC Nr." value={emp.labor_card_number} />
                <Row label="Personal Nr. (MOL)" value={emp.mol_id} />
              </Section>

              <Section title="Salary & Logistics">
                {me?.is_superuser && emp.gross_salary != null && (
                  <Row label="Gross Salary" value={`${me?.currency_code || 'AED'} ${emp.gross_salary}`} />
                )}
                <Row label="Housing Camp" value={emp.camp} />
                <Row label="Transportation" value={emp.transportation} />
              </Section>
            </div>
          ) : null}
        </GlassCard>

        {/* Attendance */}
        <div>
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <Select aria-label="Range" value={filter} onChange={(v) => setFilter(v as any)} options={FILTERS} className="w-32" />
            <DatePicker aria-label="Date" value={date} onChange={setDate} className="w-40" />
          </div>

          {isLoading ? (
            <Skeleton className="h-80 rounded-3xl" />
          ) : filter === 'daily' ? (
            <DailyView id={id} data={data} />
          ) : (
            <RangeCalendar data={data} date={date} />
          )}

          {/* History (superuser) */}
          {me?.is_superuser && emp && (
            <div className="mt-4">
              <HistoryTabs id={id} />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

/** A titled group of profile rows. */
function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div>
      <p className="mb-1.5 text-[11px] font-bold uppercase tracking-wide text-fg-muted">{title}</p>
      <dl className="space-y-2">{children}</dl>
    </div>
  )
}

function Row({ label, value }: { label: string; value?: string | null }) {
  return (
    <div className="flex items-start justify-between gap-3 border-b border-black/[0.05] pb-2 last:border-0">
      <dt className="shrink-0 text-fg-faint">{label}</dt>
      <dd className="text-right font-medium text-fg">{value || '—'}</dd>
    </div>
  )
}

/** Daily slot view: Office In / Office Out with a geofence map of the pins. */
function DailyView({ id, data }: { id: number; data: any }) {
  const slots = data?.slots as Record<string, any> | undefined
  const daySiteId = data?.day_site_id as number | null | undefined
  void id

  const { data: coords } = useQuery({
    queryKey: ['site-coords', daySiteId],
    queryFn: () => getSiteCoordinates(daySiteId as number),
    enabled: !!daySiteId,
  })

  const markers = Object.values(slots ?? {})
    .filter((s: any) => s.latitude != null && s.longitude != null)
    .map((s: any) => ({ lat: s.latitude, lng: s.longitude }))

  return (
    <div className="grid gap-4 md:grid-cols-2">
      <div className="space-y-3">
        {slots && Object.entries(slots).map(([name, s]: [string, any]) => (
          <GlassCard key={name} className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-semibold text-fg">{name}</p>
                <p className="text-xs text-fg-faint">Expected {s.time_range}</p>
              </div>
              <div className="text-right">
                <p className="text-lg font-bold text-fg">{s.check_in || '—'}</p>
                {s.late_minutes > 0 && <Pill tone="outline">{s.late_minutes}m late</Pill>}
                {s.early_minutes > 0 && <Pill tone="soft">{s.early_minutes}m early</Pill>}
              </div>
            </div>
          </GlassCard>
        ))}
        {(!slots || Object.keys(slots).length === 0) && (
          <GlassCard className="p-6 text-center text-sm text-fg-muted">No attendance recorded this day.</GlassCard>
        )}
      </div>
      <GeofenceMap
        polygon={(coords?.coordinates ?? []) as any}
        markers={markers}
        height={280}
      />
    </div>
  )
}

/** Weekly/monthly calendar view from calendar_days. */
function RangeCalendar({ data, date }: { data: any; date: string }) {
  const days = (data?.calendar_days ?? []).map((d: any) => ({
    date: d.date,
    status: (d.record?.status || (d.is_on_leave ? 'leave' : '') || '').toLowerCase(),
  })).filter((d: any) => d.status)
  const [y, m] = date.split('-').map(Number)

  return (
    <GlassCard className="p-5">
      <AttendanceCalendar year={y} month={m} days={days} />
      <div className="mt-4"><CalendarLegend /></div>
    </GlassCard>
  )
}

/** Status + site history tabs. */
function HistoryTabs({ id }: { id: number }) {
  const status = useQuery({ queryKey: ['status-history', id], queryFn: () => getStatusHistory(id) })
  const sites = useQuery({ queryKey: ['site-history', id], queryFn: () => getSiteHistory(id) })

  return (
    <Tabs
      items={[
        {
          value: 'status',
          label: 'Status history',
          content: (
            <GlassCard className="p-4">
              {(status.data?.history ?? []).length === 0 ? (
                <p className="py-4 text-center text-sm text-fg-muted">No status changes.</p>
              ) : (
                <ul className="space-y-2">
                  {status.data!.history.map((h) => (
                    <li key={h.id} className="flex items-center justify-between rounded-lg bg-black/[0.02] px-3 py-2 text-sm">
                      <span className="text-fg">{h.old_status || '—'} → <span className="font-semibold">{h.new_status}</span></span>
                      <span className="text-xs text-fg-faint">{h.changed_at?.slice(0, 10)}</span>
                    </li>
                  ))}
                </ul>
              )}
            </GlassCard>
          ),
        },
        {
          value: 'site',
          label: 'Site history',
          content: (
            <GlassCard className="p-4">
              {(sites.data?.segments ?? []).length === 0 ? (
                <p className="py-4 text-center text-sm text-fg-muted">No site changes.</p>
              ) : (
                <ul className="space-y-2">
                  {sites.data!.segments.map((s) => (
                    <li key={s.id} className="flex items-center justify-between rounded-lg bg-black/[0.02] px-3 py-2 text-sm">
                      <span className="font-medium text-fg">{s.site_name || '—'} {s.is_current && <Pill tone="outline" className="ml-1">current</Pill>}</span>
                      <span className="text-xs text-fg-faint">{s.from_date} → {s.to_date || 'now'}</span>
                    </li>
                  ))}
                </ul>
              )}
            </GlassCard>
          ),
        },
      ]}
    />
  )
}
