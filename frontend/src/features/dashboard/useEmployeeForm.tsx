import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  addEmployee,
  getEmployeeForEdit,
  updateEmployee,
  EMPLOYER_CHOICES,
  SPONSOR_CHOICES,
  MASTER_STATUSES,
  CATEGORIES,
  TRANSPORT_OPTIONS,
  LEAVE_TYPES,
} from '../../lib/api/employees'
import { Select as UiSelect, DatePicker as UiDatePicker } from '../../components/ui/Field'
import { cn } from '../../lib/utils'
import type { Site } from '../../lib/api/types'

type Form = Record<string, string>
type Files = Record<string, File | null>

export const inputCls =
  'h-10 w-full rounded-xl border border-black/10 bg-white px-3 text-sm text-fg outline-none ' +
  'focus:border-black/25 focus:ring-2 focus:ring-black/10'

/** Salary field keys — gated to superusers. */
const SALARY_KEYS = [
  'gross_salary', 'basic_salary', 'accommodation_allowance', 'transport_allowance',
  'food_allowance', 'fixed_ot_allowance', 'other_allowance', 'salary_reduction', 'salary_remarks',
]

export interface EmployeeTab {
  value: string
  label: string
  content: ReactNode
}

/**
 * Shared add/edit employee form logic — state, prefill, the multipart save
 * mutation and the full set of tabbed field groups. Both the (legacy) dialog and
 * the full-page editor consume this so the fields stay in one place.
 *
 * Values live in a flat string map (persist across tab switches); files are held
 * separately and always sent as multipart so document uploads ride along. On
 * edit the salary block is superuser-only and a few tabs (insurance, EOS,
 * passport-control) are edit-only.
 */
export function useEmployeeForm({
  employeeId,
  isSuperuser,
  sites,
  onSaved,
}: {
  employeeId: number | null
  isSuperuser: boolean
  sites: Site[]
  onSaved: () => void
}) {
  const isEdit = employeeId != null
  const qc = useQueryClient()
  const [form, setForm] = useState<Form>({})
  const [files, setFiles] = useState<Files>({})
  const [error, setError] = useState<string | null>(null)

  const { data: existing, isLoading } = useQuery({
    queryKey: ['employee-edit', employeeId],
    queryFn: () => getEmployeeForEdit(employeeId as number),
    enabled: isEdit,
  })
  useEffect(() => {
    // Coerce every prefilled value to a string — the server sends some fields
    // (e.g. `site`) as numbers, but the Select options key on string ids, so a
    // numeric value would never match and would show the raw id.
    if (existing) {
      setForm(Object.fromEntries(Object.entries(existing).map(([k, v]) => [k, v == null ? '' : String(v)])))
    }
  }, [existing])

  const set = (k: string, v: string) => setForm((f) => ({ ...f, [k]: v }))
  const setFile = (k: string, f: File | null) => setFiles((s) => ({ ...s, [k]: f }))

  const save = useMutation({
    mutationFn: () => {
      const fd = new FormData()
      Object.entries(form).forEach(([k, v]) => {
        if (k.endsWith('_url') || k === 'site_name' || k === 'id') return
        if (!isSuperuser && SALARY_KEYS.includes(k)) return
        fd.append(k, v ?? '')
      })
      Object.entries(files).forEach(([k, f]) => f && fd.append(k, f))
      return isEdit ? updateEmployee(employeeId as number, fd) : addEmployee(fd)
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['employees'] })
      qc.invalidateQueries({ queryKey: ['stats'] })
      if (isEdit) qc.invalidateQueries({ queryKey: ['user-detail', employeeId] })
      onSaved()
    },
    onError: (e: any) => setError(e?.message ?? 'Could not save employee.'),
  })

  const siteOptions = useMemo(() => sites.map((s) => ({ value: String(s.id), label: s.name })), [sites])

  // Field renderers bound to this form.
  const Txt = (k: string, label: string, type = 'text') => (
    <Field label={label}>
      <input type={type} value={form[k] ?? ''} onChange={(e) => set(k, e.target.value)} className={inputCls} />
    </Field>
  )
  const Dte = (k: string, label: string) => (
    <Field label={label}>
      <UiDatePicker value={form[k] ?? ''} onChange={(v) => set(k, v)} className="w-full" placeholder="Select date" />
    </Field>
  )
  const Num = (k: string, label: string) => (
    <Field label={label}>
      <input type="number" step="0.01" value={form[k] ?? ''} onChange={(e) => set(k, e.target.value)} className={inputCls} />
    </Field>
  )
  const Sel = (k: string, label: string, options: { value: string; label: string }[]) => (
    <Field label={label}>
      <UiSelect value={form[k] ?? ''} onChange={(v) => set(k, v)} options={options} placeholder="—" className="w-full" />
    </Field>
  )
  const Area = (k: string, label: string) => (
    <Field label={label} full>
      <textarea rows={3} value={form[k] ?? ''} onChange={(e) => set(k, e.target.value)} className={cn(inputCls, 'h-auto py-2')} />
    </Field>
  )
  const FileField = (k: string, label: string) => (
    <Field label={label}>
      <div className="flex items-center gap-2">
        <input type="file" onChange={(e) => setFile(k, e.target.files?.[0] ?? null)}
          className="text-xs text-fg-muted file:mr-2 file:rounded-lg file:border file:border-black/10 file:bg-white file:px-2 file:py-1 file:text-xs file:font-semibold" />
        {form[`${k}_url`] && (
          <a href={form[`${k}_url`]} target="_blank" rel="noreferrer" className="text-xs font-semibold text-fg underline">
            View
          </a>
        )}
      </div>
    </Field>
  )
  const enumOpts = (arr: readonly string[]) => arr.map((v) => ({ value: v, label: v }))

  const tabs: EmployeeTab[] = [
    {
      value: 'employment',
      label: 'Employment',
      content: (
        <Grid>
          {Sel('site', 'Site', siteOptions)}
          {Txt('department', 'Department')}
          {Txt('position', 'Position')}
          {Txt('badge_number', 'Badge number')}
          {Txt('salary_grade', 'Salary grade')}
          {Sel('category', 'Category', enumOpts(CATEGORIES))}
          {Sel('sponsor', 'Sponsor', enumOpts(SPONSOR_CHOICES))}
          {Sel('employer', 'Employer', enumOpts(EMPLOYER_CHOICES))}
          {Dte('date_of_joining', 'Date of joining')}
          {isEdit && Dte('site_effective_from', 'Site effective from')}
          {Area('job_description', 'Job description')}
        </Grid>
      ),
    },
    {
      value: 'personal',
      label: 'Personal',
      content: (
        <Grid>
          {Txt('name', 'Full name *')}
          {Txt('email', 'Email', 'email')}
          {Txt('phone', 'Phone')}
          {Txt('nationality', 'Nationality')}
          {Txt('gender', 'Gender')}
          {Txt('marital_status', 'Marital status')}
          {Txt('religion', 'Religion')}
          {Dte('date_of_birth', 'Date of birth')}
          {Txt('camp', 'Housing camp')}
          {Sel('transportation', 'Transportation', enumOpts(TRANSPORT_OPTIONS))}
        </Grid>
      ),
    },
    {
      value: 'documents',
      label: 'Documents',
      content: (
        <Grid>
          {Txt('passport_number', 'Passport number')}
          {Dte('passport_expiry', 'Passport expiry')}
          {Txt('visa_details', 'Visa details')}
          {Dte('visa_expiry_date', 'Visa expiry')}
          {Txt('labor_card_number', 'Labour card / CEC')}
          {isEdit && Dte('labour_card_expiry', 'Labour card expiry')}
          {Txt('mol_id', 'MOL ID')}
          {FileField('passport_document', 'Passport document')}
          {FileField('visa_document', 'Visa document')}
          {FileField('labour_card_document', 'Labour card document')}
        </Grid>
      ),
    },
    ...(isSuperuser
      ? [{
          value: 'salary',
          label: 'Salary',
          content: (
            <Grid>
              {Num('gross_salary', 'Gross salary')}
              {Num('basic_salary', 'Basic salary')}
              {Num('accommodation_allowance', 'Accommodation allowance')}
              {Num('transport_allowance', 'Transport allowance')}
              {Num('food_allowance', 'Food allowance')}
              {Num('fixed_ot_allowance', 'Fixed OT allowance')}
              {Num('other_allowance', 'Other allowance')}
              {Num('salary_reduction', 'Salary reduction')}
              {Area('salary_remarks', 'Salary remarks')}
            </Grid>
          ),
        }]
      : []),
    {
      value: 'leave',
      label: 'Leave',
      content: (
        <Grid>
          {Dte('leave_approval_date', 'Last working date (before leave)')}
          {Dte('leave_start_date', 'Leave start')}
          {Dte('leave_end_date', 'Leave end')}
          {Sel('leave_type', 'Leave type', enumOpts(LEAVE_TYPES))}
          {Sel('leave_ticket_eligible', 'Ticket eligible', enumOpts(['Eligible', 'Not Eligible']))}
          {Num('leave_ticket_price', 'Ticket price')}
        </Grid>
      ),
    },
    {
      value: 'status',
      label: 'Status',
      content: (
        <Grid>
          {Sel('status', 'Employment status', enumOpts(MASTER_STATUSES))}
          {Dte('resumption_date', 'Resumption date')}
          {Dte('last_working_date', 'Last working date')}
          {Area('termination_reason', 'Termination reason')}
          {isEdit && Txt('eos_subject', 'End-of-service subject')}
          {isEdit && Dte('eos_date', 'End-of-service date')}
          {isEdit && Area('eos_note', 'End-of-service note')}
        </Grid>
      ),
    },
    ...(isEdit
      ? [{
          value: 'insurance',
          label: 'Insurance',
          content: (
            <Grid>
              {Txt('wc_insurance_name', 'WC insurance name')}
              {Sel('wc_insurance_status', 'WC status', enumOpts(['Active', 'Inactive', 'Expired']))}
              {Dte('wc_insurance_start_date', 'WC start')}
              {Dte('wc_insurance_end_date', 'WC end')}
              {Num('wc_insurance_premium_cost', 'WC premium')}
              {Txt('medical_insurance_name', 'Medical insurance name')}
              {Txt('medical_insurance_card_number', 'Medical card number')}
              {Sel('medical_insurance_status', 'Medical status', enumOpts(['Active', 'Inactive', 'Expired']))}
              {Dte('medical_insurance_start_date', 'Medical start')}
              {Dte('medical_insurance_end_date', 'Medical end')}
              {Num('medical_insurance_premium_cost', 'Medical premium')}
            </Grid>
          ),
        }]
      : []),
  ]

  return { tabs, save, isLoading, error, setError, isEdit, name: form.name ?? '' }
}

export function Grid({ children }: { children: ReactNode }) {
  return <div className="grid gap-3 sm:grid-cols-2">{children}</div>
}

export function Field({ label, children, full }: { label: string; children: ReactNode; full?: boolean }) {
  return (
    <div className={cn(full && 'sm:col-span-2')}>
      <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-fg-faint">{label}</label>
      {children}
    </div>
  )
}
