import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { UploadCloud, FileSpreadsheet, Download, AlertCircle, CheckCircle2 } from 'lucide-react'
import { bulkEditEmployees, bulkEditTemplateUrl, type BulkEditResult } from '../../lib/api/employees'
import { Dialog } from '../../components/ui/Dialog'
import { cn } from '../../lib/utils'

/**
 * Bulk-edit employees from Excel. Flow: download the pre-filled template,
 * change only the cells you want (rows are keyed on Badge ID; blank cells are
 * left unchanged), upload, then apply. Salary columns are applied only for
 * superusers server-side. On success the employee list + stats refresh.
 */
export function BulkEditWizard({ onClose }: { onClose: () => void }) {
  const qc = useQueryClient()
  const [file, setFile] = useState<File | null>(null)

  const run = useMutation({
    mutationFn: (f: File) => bulkEditEmployees(f),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['employees'] })
      qc.invalidateQueries({ queryKey: ['stats'] })
    },
  })

  const result: BulkEditResult | undefined = run.data
  const done = !!result

  return (
    <Dialog
      open
      onOpenChange={(o) => !o && onClose()}
      title="Bulk edit employees"
      description="Download the template, edit the cells you want, then upload to apply."
      className="max-w-2xl"
      footer={
        <>
          <button type="button" onClick={onClose}
            className="h-10 rounded-xl border border-black/10 px-4 text-sm font-semibold text-fg hover:bg-black/[0.04]">
            {done ? 'Close' : 'Cancel'}
          </button>
          {!done && (
            <button type="button" disabled={!file || run.isPending}
              onClick={() => file && run.mutate(file)}
              className="h-10 rounded-xl bg-[#14141a] px-5 text-sm font-semibold text-white hover:opacity-90 disabled:opacity-40">
              {run.isPending ? 'Applying…' : 'Apply changes'}
            </button>
          )}
        </>
      }
    >
      {done ? (
        <BulkEditResults result={result!} />
      ) : (
        <div className="space-y-4">
          {/* Step 1 — template */}
          <a href={bulkEditTemplateUrl()}
            className="flex items-center gap-2 rounded-xl border border-black/10 bg-black/[0.02] px-4 py-3 text-sm font-semibold text-fg hover:bg-black/[0.04]">
            <Download size={16} /> Download the current employees as an Excel template
          </a>

          {/* Step 2 — upload */}
          <label className={cn(
            'flex cursor-pointer flex-col items-center gap-2 rounded-2xl border-2 border-dashed border-black/15 px-6 py-8 text-center transition-colors hover:border-black/30',
            file && 'border-solid border-black/20 bg-black/[0.02]',
          )}>
            <input type="file" accept=".xlsx,.xls" className="hidden"
              onChange={(e) => { setFile(e.target.files?.[0] ?? null); run.reset() }} />
            {file ? (
              <>
                <FileSpreadsheet size={26} className="text-fg" />
                <p className="text-sm font-semibold text-fg">{file.name}</p>
                <p className="text-xs text-fg-faint">Click to choose a different file</p>
              </>
            ) : (
              <>
                <UploadCloud size={26} className="text-fg-muted" />
                <p className="text-sm font-semibold text-fg">Upload the edited sheet</p>
                <p className="text-xs text-fg-faint">.xlsx — rows matched by Badge ID; blank cells unchanged</p>
              </>
            )}
          </label>

          {run.isError && (
            <p className="flex items-center gap-2 rounded-xl bg-black/[0.04] px-3 py-2 text-sm text-fg">
              <AlertCircle size={15} /> {(run.error as Error).message}
            </p>
          )}
        </div>
      )}
    </Dialog>
  )
}

/** Post-apply summary: counters + a scrollable list of per-row errors. */
function BulkEditResults({ result }: { result: BulkEditResult }) {
  return (
    <div className="space-y-4">
      <div className="grid place-items-center gap-2 py-2 text-center">
        <span className="grid h-12 w-12 place-items-center rounded-2xl bg-[#14141a] text-white"><CheckCircle2 size={22} /></span>
        <p className="font-semibold text-fg">Bulk edit applied</p>
      </div>

      <div className="grid grid-cols-3 gap-3 text-center">
        <Counter label="Updated" value={result.updated_count} />
        <Counter label="Skipped" value={result.skipped_count} />
        <Counter label="Errors" value={result.errors?.length ?? 0} />
      </div>

      {result.errors && result.errors.length > 0 && (
        <div>
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-fg-faint">Errors</p>
          <div className="max-h-56 overflow-y-auto rounded-xl border border-black/10">
            <table className="w-full text-sm">
              <tbody>
                {result.errors.map((e, i) => (
                  <tr key={i} className="border-b border-black/[0.06] last:border-0">
                    <td className="px-3 py-2 text-fg-faint">Row {e.row}</td>
                    <td className="px-3 py-2 font-medium text-fg">{e.badge_number || '—'}</td>
                    <td className="px-3 py-2 text-fg-muted">{e.error}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

function Counter({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-xl bg-black/[0.03] px-3 py-3">
      <p className="text-2xl font-bold leading-none text-fg tabular-nums">{value}</p>
      <p className="mt-1 text-xs text-fg-faint">{label}</p>
    </div>
  )
}
