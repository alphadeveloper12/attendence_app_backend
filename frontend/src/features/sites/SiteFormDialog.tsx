import { useState, type ReactNode } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { AlertCircle, MapPin } from 'lucide-react'
import { addSite, editSite, type SiteFormValues, type SiteRow } from '../../lib/api/sites'
import { Dialog } from '../../components/ui/Dialog'
import { TimePicker } from '../../components/ui/Field'
import { cn } from '../../lib/utils'

const inputCls =
  'h-10 w-full rounded-xl border border-black/10 bg-white px-3 text-sm text-fg outline-none ' +
  'focus:border-black/25 focus:ring-2 focus:ring-black/10'

/** Seed the form from an existing row (edit) or sensible defaults (add). */
function seed(site: SiteRow | null): SiteFormValues {
  return {
    name: site?.name ?? '',
    office_start: site?.office_start ?? '09:00',
    office_end: site?.office_end ?? '18:00',
    worker_start: site?.worker_start ?? '08:00',
    worker_end: site?.worker_end ?? '17:00',
    office_day_off: site?.office_day_off ?? 'Sunday',
    worker_day_off: site?.worker_day_off ?? 'Sunday',
    kml_file: null,
  }
}

/**
 * Add / edit a site — name, office & worker schedules and day-offs, plus an
 * optional `.kml` upload that (re)draws the geofence polygon. On save the sites
 * list refreshes.
 */
export function SiteFormDialog({ site, onClose }: { site: SiteRow | null; onClose: () => void }) {
  const qc = useQueryClient()
  const isEdit = !!site
  const [v, setV] = useState<SiteFormValues>(() => seed(site))

  const save = useMutation({
    mutationFn: () => (isEdit ? editSite(site!.id, v) : addSite(v)),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['sites'] })
      onClose()
    },
  })

  const set = <K extends keyof SiteFormValues>(k: K, val: SiteFormValues[K]) => setV((s) => ({ ...s, [k]: val }))

  return (
    <Dialog
      open
      onOpenChange={(o) => !o && onClose()}
      title={isEdit ? 'Edit site' : 'Add site'}
      description={isEdit ? site!.name : 'Create a new site with its schedule and geofence.'}
      className="max-w-xl"
      footer={
        <>
          <button type="button" onClick={onClose}
            className="h-10 rounded-xl border border-black/10 px-4 text-sm font-semibold text-fg hover:bg-black/[0.04]">
            Cancel
          </button>
          <button type="button" disabled={!v.name.trim() || save.isPending}
            onClick={() => save.mutate()}
            className="h-10 rounded-xl bg-[#14141a] px-5 text-sm font-semibold text-white hover:opacity-90 disabled:opacity-40">
            {save.isPending ? 'Saving…' : isEdit ? 'Save changes' : 'Add site'}
          </button>
        </>
      }
    >
      <div className="space-y-4">
        <L label="Site name">
          <input value={v.name} onChange={(e) => set('name', e.target.value)} className={inputCls} placeholder="e.g. Opal Garden" />
        </L>

        <div className="grid gap-4 sm:grid-cols-2">
          <L label="Office start"><TimePicker value={v.office_start} onChange={(t) => set('office_start', t)} className="w-full" /></L>
          <L label="Office end"><TimePicker value={v.office_end} onChange={(t) => set('office_end', t)} className="w-full" /></L>
          <L label="Worker start"><TimePicker value={v.worker_start} onChange={(t) => set('worker_start', t)} className="w-full" /></L>
          <L label="Worker end"><TimePicker value={v.worker_end} onChange={(t) => set('worker_end', t)} className="w-full" /></L>
          <L label="Office day off">
            <input value={v.office_day_off} onChange={(e) => set('office_day_off', e.target.value)} className={inputCls} placeholder="Sunday" />
          </L>
          <L label="Worker day off">
            <input value={v.worker_day_off} onChange={(e) => set('worker_day_off', e.target.value)} className={inputCls} placeholder="Sunday" />
          </L>
        </div>

        <L label="Geofence (KML)">
          <label className="flex cursor-pointer items-center gap-2 rounded-xl border border-dashed border-black/15 px-3 py-2.5 text-sm text-fg-muted hover:border-black/30">
            <MapPin size={15} />
            <input type="file" accept=".kml" className="hidden"
              onChange={(e) => set('kml_file', e.target.files?.[0] ?? null)} />
            {v.kml_file ? <span className="font-medium text-fg">{v.kml_file.name}</span>
              : isEdit && site!.geofence_filename ? <span>Current: {site!.geofence_filename} — upload to replace</span>
              : <span>Upload a .kml file to set the geofence (optional)</span>}
          </label>
        </L>

        {save.isError && (
          <p className="flex items-center gap-2 rounded-xl bg-black/[0.04] px-3 py-2 text-sm text-fg">
            <AlertCircle size={15} /> {(save.error as Error).message}
          </p>
        )}
      </div>
    </Dialog>
  )
}

function L({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div>
      <label className={cn('mb-1 block text-xs font-semibold uppercase tracking-wide text-fg-faint')}>{label}</label>
      {children}
    </div>
  )
}
