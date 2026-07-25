import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { UploadCloud, FileSpreadsheet, Check, AlertCircle } from 'lucide-react'
import { importPreview, importEmployees, type ImportColumn } from '../../lib/api/employees'
import { Dialog } from '../../components/ui/Dialog'
import { Pill } from '../../components/ui/Pill'
import { cn } from '../../lib/utils'

/** Confidence status → pill tone. */
const statusTone: Record<string, 'solid' | 'outline' | 'soft'> = {
  strong: 'solid',
  likely: 'outline',
  weak: 'soft',
  unknown: 'soft',
}

/**
 * Import employees from Excel — a 3-step wizard: pick a file, review the
 * auto-detected column mapping (informational; the server maps on its own), then
 * confirm the import. On success the employee list + stats refresh.
 */
export function ImportWizard({ onClose }: { onClose: () => void }) {
  const qc = useQueryClient()
  const [file, setFile] = useState<File | null>(null)
  const [done, setDone] = useState<string | null>(null)

  const preview = useMutation({ mutationFn: (f: File) => importPreview(f) })
  const run = useMutation({
    mutationFn: (f: File) => importEmployees(f),
    onSuccess: (res) => {
      qc.invalidateQueries({ queryKey: ['employees'] })
      qc.invalidateQueries({ queryKey: ['stats'] })
      const msg =
        (res.message as string) ||
        (res.imported != null ? `Imported ${res.imported} employees.` : 'Import complete.')
      setDone(String(msg))
    },
  })

  function pick(f: File | null) {
    setFile(f)
    setDone(null)
    if (f) preview.mutate(f)
  }

  const cols = preview.data?.columns ?? []

  return (
    <Dialog
      open
      onOpenChange={(o) => !o && onClose()}
      title="Import employees"
      description="Upload an Excel sheet — columns are auto-detected."
      className="max-w-2xl"
      footer={
        <>
          <button type="button" onClick={onClose}
            className="h-10 rounded-xl border border-black/10 px-4 text-sm font-semibold text-fg hover:bg-black/[0.04]">
            {done ? 'Close' : 'Cancel'}
          </button>
          {!done && (
            <button type="button" disabled={!file || run.isPending || preview.isPending}
              onClick={() => file && run.mutate(file)}
              className="h-10 rounded-xl bg-[#14141a] px-5 text-sm font-semibold text-white hover:opacity-90 disabled:opacity-40">
              {run.isPending ? 'Importing…' : 'Import'}
            </button>
          )}
        </>
      }
    >
      {done ? (
        <div className="grid place-items-center gap-2 py-8 text-center">
          <span className="grid h-12 w-12 place-items-center rounded-2xl bg-[#14141a] text-white"><Check size={22} /></span>
          <p className="font-semibold text-fg">{done}</p>
        </div>
      ) : (
        <div className="space-y-4">
          {/* File picker */}
          <label className={cn(
            'flex cursor-pointer flex-col items-center gap-2 rounded-2xl border-2 border-dashed border-black/15 px-6 py-8 text-center transition-colors hover:border-black/30',
            file && 'border-solid border-black/20 bg-black/[0.02]',
          )}>
            <input type="file" accept=".xlsx,.xls" className="hidden"
              onChange={(e) => pick(e.target.files?.[0] ?? null)} />
            {file ? (
              <>
                <FileSpreadsheet size={26} className="text-fg" />
                <p className="text-sm font-semibold text-fg">{file.name}</p>
                <p className="text-xs text-fg-faint">Click to choose a different file</p>
              </>
            ) : (
              <>
                <UploadCloud size={26} className="text-fg-muted" />
                <p className="text-sm font-semibold text-fg">Choose an Excel file</p>
                <p className="text-xs text-fg-faint">.xlsx or .xls</p>
              </>
            )}
          </label>

          {preview.isPending && <p className="text-center text-sm text-fg-muted">Detecting columns…</p>}
          {preview.isError && (
            <p className="flex items-center gap-2 rounded-xl bg-black/[0.04] px-3 py-2 text-sm text-fg">
              <AlertCircle size={15} /> {(preview.error as Error).message}
            </p>
          )}

          {/* Mapping preview */}
          {cols.length > 0 && (
            <div>
              <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-fg-faint">
                Detected mapping ({cols.filter((c) => c.matched_field).length}/{cols.length} matched)
              </p>
              <div className="max-h-64 overflow-y-auto rounded-xl border border-black/10">
                <table className="w-full text-sm">
                  <tbody>
                    {cols.map((c: ImportColumn) => (
                      <tr key={c.column_index} className="border-b border-black/[0.06] last:border-0">
                        <td className="px-3 py-2 text-fg-muted">{c.header || `Column ${c.column_index + 1}`}</td>
                        <td className="px-3 py-2 font-medium text-fg">{c.matched_field ?? <span className="text-fg-faint">— unmapped —</span>}</td>
                        <td className="px-3 py-2 text-right">
                          <Pill tone={statusTone[c.status] ?? 'soft'}>{c.confidence}%</Pill>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

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
