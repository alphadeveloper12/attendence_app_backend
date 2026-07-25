import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient, keepPreviousData } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { ChevronLeft, ChevronRight, Trash2, Download, Pencil } from 'lucide-react'
import {
  getEmployees,
  bulkDeleteEmployees,
  type EmployeeRow,
  type EmployeeListParams,
} from '../../lib/api/employees'
import { GlassCard } from '../../components/ui/GlassCard'
import { EmptyState } from '../../components/ui/EmptyState'
import { SkeletonRows } from '../../components/ui/Skeleton'
import { Pill } from '../../components/ui/Pill'
import { downloadPost } from '../../lib/download'
import { cn } from '../../lib/utils'

/** Status → pill tone (monochrome). Active reads as solid, exits as soft. */
function statusTone(status: string | null): 'solid' | 'outline' | 'soft' {
  if (status === 'Active') return 'solid'
  if (status === 'Leave') return 'outline'
  return 'soft'
}

/**
 * The main workforce table: server-paginated + filtered, with row selection and
 * bulk actions (delete for superusers, export selected). Row click opens the
 * employee detail; the edit pencil opens the edit form via `onEdit`.
 */
export function EmployeeTable({
  filters,
  canWrite,
  onEdit,
}: {
  filters: EmployeeListParams
  canWrite: boolean
  onEdit?: (id: number) => void
}) {
  const [page, setPage] = useState(1)
  const [selected, setSelected] = useState<Set<number>>(new Set())
  const navigate = useNavigate()
  const qc = useQueryClient()

  // Reset to page 1 whenever filters change (via a stable key).
  const filterKey = JSON.stringify(filters)
  const { data, isLoading, isFetching } = useQuery({
    queryKey: ['employees', filterKey, page],
    queryFn: () => getEmployees({ ...filters, page, per_page: 25 }),
    placeholderData: keepPreviousData,
  })

  const del = useMutation({
    mutationFn: (ids: number[]) => bulkDeleteEmployees(ids),
    onSuccess: () => {
      setSelected(new Set())
      qc.invalidateQueries({ queryKey: ['employees'] })
      qc.invalidateQueries({ queryKey: ['stats'] })
    },
  })

  const rows = data?.results ?? []
  const total = data?.count ?? 0
  const perPage = 25
  const totalPages = Math.max(1, Math.ceil(total / perPage))

  const allOnPageSelected = rows.length > 0 && rows.every((r) => selected.has(r.id))
  function toggleAll() {
    setSelected((prev) => {
      const next = new Set(prev)
      if (allOnPageSelected) rows.forEach((r) => next.delete(r.id))
      else rows.forEach((r) => next.add(r.id))
      return next
    })
  }
  function toggle(id: number) {
    setSelected((prev) => {
      const next = new Set(prev)
      next.has(id) ? next.delete(id) : next.add(id)
      return next
    })
  }

  const selectedIds = useMemo(() => Array.from(selected), [selected])

  return (
    <GlassCard className="overflow-hidden">
      {/* Bulk action bar (only when something is selected) */}
      {selectedIds.length > 0 && (
        <div className="flex items-center justify-between gap-3 border-b border-black/10 bg-black/[0.02] px-4 py-2.5">
          <span className="text-sm font-medium text-fg">{selectedIds.length} selected</span>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() =>
                downloadPost('/employees/export-selected/', { ids: selectedIds }).catch(() => {})
              }
              className="flex h-8 items-center gap-1.5 rounded-lg border border-black/10 px-3 text-xs font-semibold text-fg hover:bg-black/[0.04]"
            >
              <Download size={13} /> Export
            </button>
            {canWrite && (
              <button
                type="button"
                onClick={() => {
                  if (confirm(`Delete ${selectedIds.length} employees? This cannot be undone.`))
                    del.mutate(selectedIds)
                }}
                className="flex h-8 items-center gap-1.5 rounded-lg bg-[#14141a] px-3 text-xs font-semibold text-white hover:opacity-90"
              >
                <Trash2 size={13} /> Delete
              </button>
            )}
          </div>
        </div>
      )}

      {isLoading ? (
        <div className="p-4">
          <SkeletonRows rows={10} />
        </div>
      ) : rows.length === 0 ? (
        <EmptyState message="No employees match the current filters." />
      ) : (
        <div className={cn('overflow-x-auto', isFetching && 'opacity-60')}>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-black/10 text-xs font-semibold uppercase tracking-wide text-fg-faint">
                <th className="w-10 px-4 py-3">
                  <input
                    type="checkbox"
                    checked={allOnPageSelected}
                    onChange={toggleAll}
                    className="h-4 w-4 accent-[#14141a]"
                    aria-label="Select all on page"
                  />
                </th>
                <th className="px-4 py-3 text-left">Employee</th>
                <th className="px-4 py-3 text-left">Site</th>
                <th className="px-4 py-3 text-left">Department</th>
                <th className="px-4 py-3 text-left">Position</th>
                <th className="px-4 py-3 text-left">Status</th>
                {canWrite && <th className="w-12 px-4 py-3" />}
              </tr>
            </thead>
            <tbody>
              {rows.map((e: EmployeeRow) => (
                <tr
                  key={e.id}
                  className="border-b border-black/[0.06] transition-colors last:border-0 hover:bg-black/[0.02]"
                >
                  <td className="px-4 py-3">
                    <input
                      type="checkbox"
                      checked={selected.has(e.id)}
                      onChange={() => toggle(e.id)}
                      className="h-4 w-4 accent-[#14141a]"
                      aria-label={`Select ${e.name}`}
                    />
                  </td>
                  <td
                    className="cursor-pointer px-4 py-3"
                    onClick={() => navigate(`/dashboard/user/${e.id}-v2`)}
                  >
                    <div className="flex items-center gap-2.5">
                      {e.profile_picture_url ? (
                        <img src={e.profile_picture_url} alt="" className="h-8 w-8 rounded-full object-cover" />
                      ) : (
                        <span className="grid h-8 w-8 place-items-center rounded-full bg-black/[0.06] text-[11px] font-bold text-fg-muted">
                          {e.name.slice(0, 2).toUpperCase()}
                        </span>
                      )}
                      <div>
                        <p className="font-semibold text-fg">{e.name}</p>
                        <p className="text-xs text-fg-faint">#{e.badge_number || '—'}</p>
                      </div>
                    </div>
                  </td>
                  <td className="px-4 py-3 text-fg-muted">{e.site_name || '—'}</td>
                  <td className="px-4 py-3 text-fg-muted">{e.department || '—'}</td>
                  <td className="px-4 py-3 text-fg-muted">{e.position || e.salary_grade || '—'}</td>
                  <td className="px-4 py-3">
                    <Pill tone={statusTone(e.status)}>{e.status || '—'}</Pill>
                  </td>
                  {canWrite && (
                    <td className="px-4 py-3">
                      <button
                        type="button"
                        onClick={() => onEdit?.(e.id)}
                        className="grid h-8 w-8 place-items-center rounded-lg text-fg-muted hover:bg-black/[0.05] hover:text-fg"
                        title="Edit"
                      >
                        <Pencil size={15} />
                      </button>
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Pagination */}
      {total > perPage && (
        <div className="flex items-center justify-between border-t border-black/10 px-4 py-3 text-sm text-fg-muted">
          <span>
            {((page - 1) * perPage + 1).toLocaleString()}–{Math.min(page * perPage, total).toLocaleString()} of{' '}
            {total.toLocaleString()}
          </span>
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page <= 1}
              className="grid h-8 w-8 place-items-center rounded-lg hover:bg-black/[0.05] hover:text-fg disabled:opacity-40"
            >
              <ChevronLeft size={16} />
            </button>
            <span className="px-2 text-xs">Page {page} / {totalPages}</span>
            <button
              type="button"
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page >= totalPages}
              className="grid h-8 w-8 place-items-center rounded-lg hover:bg-black/[0.05] hover:text-fg disabled:opacity-40"
            >
              <ChevronRight size={16} />
            </button>
          </div>
        </div>
      )}
    </GlassCard>
  )
}
