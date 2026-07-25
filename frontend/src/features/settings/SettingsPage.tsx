import { useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Save, Plus, Trash2, Check } from 'lucide-react'
import {
  getSettings,
  updateSettings,
  getHolidays,
  createHoliday,
  deleteHoliday,
  type AppSettings,
} from '../../lib/api/settings'
import { PageHeader } from '../../components/ui/PageHeader'
import { GlassCard } from '../../components/ui/GlassCard'
import { Skeleton } from '../../components/ui/Skeleton'
import { Tabs } from '../../components/ui/Tabs'
import { DatePicker, TimePicker } from '../../components/ui/Field'
import { cn } from '../../lib/utils'

const inputCls =
  'h-10 w-full rounded-xl border border-black/10 bg-white px-3 text-sm text-fg outline-none ' +
  'focus:border-black/25 focus:ring-2 focus:ring-black/10'

/** Friendly labels for the nav-visibility feature flags. */
const NAV_LABELS: Record<string, string> = {
  dashboard: 'Dashboard',
  reports: 'Reports',
  monthly_report: 'Monthly Report',
  distribution_list: 'Distribution',
  departments: 'Departments',
  user_face: 'Face Enrollment',
  attrition_risk: 'Attrition Risk',
  document_expiry: 'Document Expiry',
  manpower_recs: 'Manpower Recs',
  ask_data: 'Ask the Data',
  geofence_tuning: 'Geofence Tuning',
  salary: 'Salary',
  sites: 'Sites',
  site_admins: 'Site Admins',
  settings: 'Settings',
}

/**
 * Settings — the global app configuration (superuser only). Grouped into tabs;
 * edits are held locally and saved as a single PUT (only changed keys matter,
 * the backend applies provided keys). Public holidays live on their own
 * endpoint with immediate add/delete.
 */
export default function SettingsPage() {
  const qc = useQueryClient()
  const { data, isLoading } = useQuery({ queryKey: ['settings'], queryFn: getSettings })

  // Local editable copy, seeded once the settings load.
  const [form, setForm] = useState<AppSettings | null>(null)
  useEffect(() => {
    if (data && !form) setForm(data)
  }, [data, form])

  const save = useMutation({
    mutationFn: (patch: Partial<AppSettings>) => updateSettings(patch),
    onSuccess: (res) => {
      qc.setQueryData(['settings'], res.settings)
      setForm(res.settings)
    },
  })

  if (isLoading || !form) {
    return (
      <div>
        <PageHeader title="Settings" subtitle="Loading…" />
        <Skeleton className="h-96 rounded-3xl" />
      </div>
    )
  }

  // Typed setter for any settings field.
  const set = <K extends keyof AppSettings>(key: K, value: AppSettings[K]) =>
    setForm((f) => (f ? { ...f, [key]: value } : f))

  return (
    <div>
      <PageHeader
        title="Settings"
        subtitle={form.updated_at ? `Last updated ${form.updated_at}` : 'Global configuration'}
        actions={
          <button
            type="button"
            disabled={save.isPending}
            onClick={() => save.mutate(form)}
            className="flex h-10 items-center gap-1.5 rounded-xl bg-[#14141a] px-4 text-sm font-semibold text-white hover:opacity-90 disabled:opacity-40"
          >
            {save.isSuccess && !save.isPending ? <Check size={15} /> : <Save size={15} />}
            {save.isPending ? 'Saving…' : save.isSuccess ? 'Saved' : 'Save changes'}
          </button>
        }
      />

      <Tabs
        items={[
          { value: 'attendance', label: 'Attendance', content: <AttendanceTab form={form} set={set} /> },
          { value: 'geofence', label: 'Geofence', content: <GeofenceTab form={form} set={set} /> },
          { value: 'salary', label: 'Salary', content: <SalaryTab form={form} set={set} /> },
          { value: 'documents', label: 'Documents', content: <DocumentsTab form={form} set={set} /> },
          { value: 'lists', label: 'Master lists', content: <ListsTab form={form} set={set} /> },
          { value: 'navigation', label: 'Navigation', content: <NavTab form={form} set={set} /> },
          { value: 'holidays', label: 'Holidays', content: <HolidaysTab /> },
        ]}
      />
    </div>
  )
}

// ---- Shared field helpers ----

type SetFn = <K extends keyof AppSettings>(key: K, value: AppSettings[K]) => void
interface TabProps {
  form: AppSettings
  set: SetFn
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-fg-faint">{label}</label>
      {children}
    </div>
  )
}

/** A labelled numeric input bound to a settings key. */
function NumField({ form, set, k, label }: TabProps & { k: keyof AppSettings; label: string }) {
  return (
    <Field label={label}>
      <input
        type="number"
        value={String(form[k] ?? '')}
        onChange={(e) => set(k, Number(e.target.value) as AppSettings[typeof k])}
        className={inputCls}
      />
    </Field>
  )
}

/** A labelled time input (HH:MM) bound to a settings key. */
function TimeField({ form, set, k, label }: TabProps & { k: keyof AppSettings; label: string }) {
  return (
    <Field label={label}>
      <TimePicker value={String(form[k] ?? '')} onChange={(v) => set(k, v as AppSettings[typeof k])} className="w-full" />
    </Field>
  )
}

/** A textarea editing a string[] as one item per line. */
function ListField({ form, set, k, label }: TabProps & { k: keyof AppSettings; label: string }) {
  const value = (form[k] as string[]) ?? []
  return (
    <Field label={label}>
      <textarea
        rows={5}
        value={value.join('\n')}
        onChange={(e) =>
          set(k, e.target.value.split('\n').map((s) => s.trim()).filter(Boolean) as AppSettings[typeof k])
        }
        className={cn(inputCls, 'h-auto py-2 font-mono text-xs leading-relaxed')}
        placeholder="One per line"
      />
    </Field>
  )
}

// ---- Tabs ----

function AttendanceTab({ form, set }: TabProps) {
  return (
    <GlassCard className="p-5">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <TimeField form={form} set={set} k="default_office_start_time" label="Office start" />
        <TimeField form={form} set={set} k="default_office_end_time" label="Office end" />
        <Field label="Office day off">
          <input value={form.default_office_day_off} onChange={(e) => set('default_office_day_off', e.target.value)} className={inputCls} />
        </Field>
        <TimeField form={form} set={set} k="default_worker_start_time" label="Worker start" />
        <TimeField form={form} set={set} k="default_worker_end_time" label="Worker end" />
        <Field label="Worker day off">
          <input value={form.default_worker_day_off} onChange={(e) => set('default_worker_day_off', e.target.value)} className={inputCls} />
        </Field>
        <NumField form={form} set={set} k="late_grace_minutes" label="Late grace (min)" />
        <NumField form={form} set={set} k="half_day_threshold_hours" label="Half-day threshold (hrs)" />
        <NumField form={form} set={set} k="normal_ot_threshold_minutes" label="OT threshold (min)" />
      </div>
      <div className="mt-4">
        <ListField form={form} set={set} k="weekend_days" label="Weekend days" />
      </div>
    </GlassCard>
  )
}

function GeofenceTab({ form, set }: TabProps) {
  return (
    <GlassCard className="p-5">
      <div className="grid gap-4 sm:grid-cols-2">
        <NumField form={form} set={set} k="default_geofence_radius_meters" label="Default radius (m)" />
        <NumField form={form} set={set} k="gps_accuracy_tolerance_meters" label="GPS tolerance (m)" />
      </div>
    </GlassCard>
  )
}

/** Gross-salary formula components (key → label), in the legacy order. */
const GROSS_COMPONENTS: { key: string; label: string }[] = [
  { key: 'basic_salary', label: 'Basic Salary' },
  { key: 'accommodation_allowance', label: 'Accommodation / CCA' },
  { key: 'transport_allowance', label: 'Transport Allowance' },
  { key: 'food_allowance', label: 'Food Allowance' },
  { key: 'fixed_ot_allowance', label: 'Fixed OT Allowance' },
  { key: 'other_allowance', label: 'Others' },
  { key: 'salary_reduction', label: 'Salary Reduction' },
]
const SIGN_OPTS = [
  { value: 1, label: 'Add' },
  { value: 0, label: 'Ignore' },
  { value: -1, label: 'Subtract' },
]

function SalaryTab({ form, set }: TabProps) {
  const formula = form.gross_formula ?? {}
  const setSign = (key: string, sign: number) => set('gross_formula', { ...formula, [key]: sign })

  return (
    <div className="space-y-4">
      <GlassCard className="p-5">
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Currency code">
            <input value={form.currency_code} onChange={(e) => set('currency_code', e.target.value)} className={inputCls} />
          </Field>
        </div>
        <label className="mt-4 flex cursor-pointer items-center gap-3">
          <Toggle checked={form.salary_superuser_only} onChange={(v) => set('salary_superuser_only', v)} />
          <span className="text-sm font-medium text-fg">Restrict salary data to superusers</span>
        </label>
      </GlassCard>

      {/* Gross salary formula — how each pay component contributes to gross. */}
      <GlassCard className="p-5">
        <h3 className="mb-1 text-sm font-bold text-fg">Gross salary formula</h3>
        <p className="mb-4 text-xs text-fg-muted">
          Choose how each component contributes to the gross salary: add it, subtract it, or ignore it.
        </p>
        <div className="space-y-2">
          {GROSS_COMPONENTS.map(({ key, label }) => (
            <div key={key} className="flex flex-wrap items-center justify-between gap-2 rounded-xl bg-black/[0.03] px-3 py-2.5">
              <span className="text-sm font-medium text-fg">{label}</span>
              <div className="inline-flex rounded-lg bg-white/70 p-0.5">
                {SIGN_OPTS.map((o) => {
                  const active = (formula[key] ?? 0) === o.value
                  return (
                    <button key={o.value} type="button" onClick={() => setSign(key, o.value)}
                      className={cn('rounded-md px-3 py-1 text-xs font-semibold transition-all',
                        active ? 'bg-[#14141a] text-white shadow-sm' : 'text-fg-muted hover:text-fg')}>
                      {o.label}
                    </button>
                  )
                })}
              </div>
            </div>
          ))}
        </div>
      </GlassCard>
    </div>
  )
}

function DocumentsTab({ form, set }: TabProps) {
  return (
    <GlassCard className="p-5">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <NumField form={form} set={set} k="passport_reminder_lead_days" label="Passport lead (days)" />
        <NumField form={form} set={set} k="visa_reminder_lead_days" label="Visa lead (days)" />
        <NumField form={form} set={set} k="labour_card_reminder_lead_days" label="Labour card lead (days)" />
        <NumField form={form} set={set} k="mol_reminder_lead_days" label="MOL lead (days)" />
      </div>
      <div className="mt-4">
        <ListField form={form} set={set} k="expiry_alert_recipients" label="Expiry alert recipients (emails)" />
      </div>
      <div className="mt-4 sm:w-1/2">
        <NumField form={form} set={set} k="distribution_snapshot_retention_months" label="Snapshot retention (months, 0 = forever)" />
      </div>
    </GlassCard>
  )
}

function ListsTab({ form, set }: TabProps) {
  return (
    <GlassCard className="p-5">
      <div className="grid gap-4 sm:grid-cols-2">
        <ListField form={form} set={set} k="sponsors" label="Sponsors" />
        <ListField form={form} set={set} k="employers" label="Employers" />
      </div>
    </GlassCard>
  )
}

function NavTab({ form, set }: TabProps) {
  const nav = form.nav_visibility ?? {}
  const keys = Object.keys(nav)
  return (
    <GlassCard className="p-5">
      <p className="mb-4 text-sm text-fg-muted">
        Toggle which pages appear in the sidebar. Settings is always superuser-only.
      </p>
      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
        {keys.map((k) => (
          <label key={k} className="flex cursor-pointer items-center justify-between rounded-xl bg-black/[0.03] px-3 py-2.5">
            <span className="text-sm font-medium text-fg">{NAV_LABELS[k] ?? k}</span>
            <Toggle
              checked={!!nav[k]}
              onChange={(v) => set('nav_visibility', { ...nav, [k]: v })}
            />
          </label>
        ))}
      </div>
    </GlassCard>
  )
}

function HolidaysTab() {
  const qc = useQueryClient()
  const { data } = useQuery({ queryKey: ['holidays'], queryFn: getHolidays })
  const refresh = () => qc.invalidateQueries({ queryKey: ['holidays'] })
  const add = useMutation({ mutationFn: createHoliday, onSuccess: refresh })
  const del = useMutation({ mutationFn: deleteHoliday, onSuccess: refresh })

  const [name, setName] = useState('')
  const [date, setDate] = useState('')
  const [recurring, setRecurring] = useState(false)
  const holidays = data?.holidays ?? []

  return (
    <GlassCard className="p-5">
      <form
        onSubmit={(e) => {
          e.preventDefault()
          if (!name.trim() || !date) return
          add.mutate({ name: name.trim(), date, recurring_annually: recurring },
            { onSuccess: () => { setName(''); setDate(''); setRecurring(false) } })
        }}
        className="mb-4 flex flex-col gap-2 sm:flex-row sm:items-center"
      >
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Holiday name" className={cn(inputCls, 'sm:flex-1')} />
        <DatePicker value={date} onChange={setDate} className="sm:w-44" />
        <label className="flex items-center gap-2 px-1 text-sm text-fg-muted">
          <Toggle checked={recurring} onChange={setRecurring} />
          Annual
        </label>
        <button type="submit" disabled={add.isPending || !name.trim() || !date}
          className="flex h-10 shrink-0 items-center gap-1.5 rounded-xl bg-[#14141a] px-4 text-sm font-semibold text-white hover:opacity-90 disabled:opacity-40">
          <Plus size={15} /> Add
        </button>
      </form>

      {holidays.length === 0 ? (
        <p className="py-6 text-center text-sm text-fg-muted">No holidays configured.</p>
      ) : (
        <ul className="divide-y divide-black/[0.06]">
          {holidays.map((h) => (
            <li key={h.id} className="flex items-center justify-between py-2.5">
              <div>
                <p className="text-sm font-semibold text-fg">{h.name}</p>
                <p className="text-xs text-fg-faint">
                  {h.date}{h.recurring_annually ? ' · repeats annually' : ''}
                </p>
              </div>
              <button type="button" onClick={() => del.mutate(h.id)}
                className="grid h-8 w-8 place-items-center rounded-lg text-fg-muted hover:bg-black/[0.05] hover:text-fg">
                <Trash2 size={15} />
              </button>
            </li>
          ))}
        </ul>
      )}
    </GlassCard>
  )
}

/** Minimal monochrome toggle switch. */
function Toggle({ checked, onChange }: { checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className={cn(
        // flex + px-0.5 keeps the knob vertically centred and inside the track;
        // a standard translate-x-5 (not an arbitrary value) lands it flush right.
        'inline-flex h-6 w-11 shrink-0 items-center rounded-full px-0.5 transition-colors',
        checked ? 'bg-[#14141a]' : 'bg-black/15',
      )}
    >
      <span
        className={cn(
          'h-5 w-5 rounded-full bg-white shadow transition-transform',
          checked ? 'translate-x-5' : 'translate-x-0',
        )}
      />
    </button>
  )
}
