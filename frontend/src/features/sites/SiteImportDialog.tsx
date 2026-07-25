import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { UploadCloud, FileSpreadsheet, Download, AlertCircle, Check } from 'lucide-react'
import { importSites, importSiteSchedule, siteScheduleTemplateUrl } from '../../lib/api/sites'
import { Dialog } from '../../components/ui/Dialog'
import { Tabs } from '../../components/ui/Tabs'
import { cn } from '../../lib/utils'

/**
 * Import sites from Excel — two modes matching the legacy page:
 *   • New sites   — bulk-create by name.
 *   • Schedule    — upsert day-offs + duty times (with a template to download).
 * On success the sites list refreshes.
 */
export function SiteImportDialog({ onClose }: { onClose: () => void }) {
  return (
    <Dialog open onOpenChange={(o) => !o && onClose()} title="Import sites"
      description="Bulk-create sites or upload their schedules from Excel." className="max-w-lg">
      <Tabs
        items={[
          { value: 'new', label: 'New sites', content: <ImportPane kind="new" onClose={onClose} /> },
          { value: 'schedule', label: 'Site schedule', content: <ImportPane kind="schedule" onClose={onClose} /> },
        ]}
      />
    </Dialog>
  )
}

function ImportPane({ kind, onClose }: { kind: 'new' | 'schedule'; onClose: () => void }) {
  const qc = useQueryClient()
  const [file, setFile] = useState<File | null>(null)

  const run = useMutation({
    mutationFn: (f: File) => (kind === 'new' ? importSites(f) : importSiteSchedule(f)),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['sites'] }),
  })
  const res = run.data as { imported_count?: number; errors?: string[] } | undefined

  return (
    <div className="space-y-4 pt-2">
      {kind === 'schedule' && (
        <a href={siteScheduleTemplateUrl()}
          className="flex items-center gap-2 rounded-xl border border-black/10 bg-black/[0.02] px-4 py-2.5 text-sm font-semibold text-fg hover:bg-black/[0.04]">
          <Download size={15} /> Download the Excel template
        </a>
      )}

      {res ? (
        <div className="space-y-3">
          <div className="grid place-items-center gap-2 py-2 text-center">
            <span className="grid h-11 w-11 place-items-center rounded-2xl bg-[#14141a] text-white"><Check size={20} /></span>
            <p className="font-semibold text-fg">Imported {res.imported_count ?? 0} site{res.imported_count === 1 ? '' : 's'}.</p>
          </div>
          {res.errors && res.errors.length > 0 && (
            <ul className="max-h-40 space-y-1 overflow-y-auto rounded-xl border border-black/10 p-2 text-xs text-fg-muted">
              {res.errors.map((e, i) => <li key={i}>• {e}</li>)}
            </ul>
          )}
          <button type="button" onClick={onClose}
            className="h-10 w-full rounded-xl border border-black/10 text-sm font-semibold text-fg hover:bg-black/[0.04]">
            Done
          </button>
        </div>
      ) : (
        <>
          <label className={cn(
            'flex cursor-pointer flex-col items-center gap-2 rounded-2xl border-2 border-dashed border-black/15 px-6 py-8 text-center transition-colors hover:border-black/30',
            file && 'border-solid border-black/20 bg-black/[0.02]',
          )}>
            <input type="file" accept=".xlsx,.xls" className="hidden" onChange={(e) => { setFile(e.target.files?.[0] ?? null); run.reset() }} />
            {file ? (
              <><FileSpreadsheet size={24} className="text-fg" /><p className="text-sm font-semibold text-fg">{file.name}</p></>
            ) : (
              <><UploadCloud size={24} className="text-fg-muted" />
                <p className="text-sm font-semibold text-fg">Choose an Excel file</p>
                <p className="text-xs text-fg-faint">{kind === 'new' ? 'One site name per row' : '.xlsx schedule sheet'}</p></>
            )}
          </label>

          {run.isError && (
            <p className="flex items-center gap-2 rounded-xl bg-black/[0.04] px-3 py-2 text-sm text-fg">
              <AlertCircle size={15} /> {(run.error as Error).message}
            </p>
          )}

          <button type="button" disabled={!file || run.isPending} onClick={() => file && run.mutate(file)}
            className="h-10 w-full rounded-xl bg-[#14141a] text-sm font-semibold text-white hover:opacity-90 disabled:opacity-40">
            {run.isPending ? 'Importing…' : 'Import'}
          </button>
        </>
      )}
    </div>
  )
}
