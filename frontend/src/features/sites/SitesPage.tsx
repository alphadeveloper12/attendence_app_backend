import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient, keepPreviousData } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { ChevronLeft, ChevronRight, MapPin, Users, Plus, Upload, Pencil, Trash2 } from 'lucide-react'
import { getSites, deleteSite, bulkDeleteSites, type SiteRow } from '../../lib/api/sites'
import { useMe } from '../../lib/hooks/useMe'
import { PageHeader } from '../../components/ui/PageHeader'
import { GlassCard } from '../../components/ui/GlassCard'
import { EmptyState } from '../../components/ui/EmptyState'
import { SkeletonRows } from '../../components/ui/Skeleton'
import { SearchInput } from '../../components/ui/Field'
import { Pill } from '../../components/ui/Pill'
import { useDebounce } from '../../lib/hooks/useDebounce'
import { SiteFormDialog } from './SiteFormDialog'
import { SiteImportDialog } from './SiteImportDialog'
import { cn } from '../../lib/utils'

/** Compact "start–end" schedule label, or "—" if unset. */
function schedule(start: string, end: string) {
  if (!start && !end) return '—'
  return `${start || '—'} – ${end || '—'}`
}

/**
 * Sites — searchable, server-paginated list with the same actions as the legacy
 * page: add / edit / delete a site, bulk-delete selected, import (new sites or
 * schedules), and open the map detail. Superusers see the write actions.
 */
export default function SitesPage() {
  const { data: me } = useMe()
  const canWrite = !!me?.can_write
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(1)
  const [selected, setSelected] = useState<Set<number>>(new Set())
  const [editing, setEditing] = useState<SiteRow | 'new' | null>(null)
  const [importing, setImporting] = useState(false)
  const navigate = useNavigate()
  const qc = useQueryClient()
  const debounced = useDebounce(search, 300)

  const { data, isLoading, isFetching } = useQuery({
    queryKey: ['sites', debounced, page],
    queryFn: () => getSites({ search: debounced || undefined, page, per_page: 12 }),
    placeholderData: keepPreviousData,
  })

  const del = useMutation({
    mutationFn: (id: number) => deleteSite(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['sites'] }),
  })
  const bulkDel = useMutation({
    mutationFn: (ids: number[]) => bulkDeleteSites(ids),
    onSuccess: () => { setSelected(new Set()); qc.invalidateQueries({ queryKey: ['sites'] }) },
  })

  const rows = data?.results ?? []
  const pg = data?.pagination
  const selectedIds = useMemo(() => Array.from(selected), [selected])
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

  return (
    <div>
      <PageHeader
        title="Sites"
        subtitle={pg ? `${pg.total_items.toLocaleString()} sites` : 'Loading…'}
        actions={
          <div className="flex flex-wrap items-center gap-2">
            <SearchInput value={search} onChange={(v) => { setSearch(v); setPage(1) }} placeholder="Search sites…" className="w-52" />
            {canWrite && (
              <button type="button" onClick={() => setImporting(true)}
                className="flex h-10 items-center gap-1.5 rounded-xl border border-black/10 bg-white/60 px-3.5 text-sm font-semibold text-fg hover:bg-white">
                <Upload size={15} /> Import
              </button>
            )}
            {canWrite && (
              <button type="button" onClick={() => setEditing('new')}
                className="flex h-10 items-center gap-1.5 rounded-xl bg-[#14141a] px-4 text-sm font-semibold text-white hover:opacity-90">
                <Plus size={15} /> Add site
              </button>
            )}
          </div>
        }
      />

      {isLoading ? (
        <GlassCard className="p-4"><SkeletonRows rows={8} /></GlassCard>
      ) : rows.length === 0 ? (
        <GlassCard><EmptyState message="No sites match your search." /></GlassCard>
      ) : (
        <GlassCard className="overflow-hidden">
          {/* Bulk action bar */}
          {canWrite && selectedIds.length > 0 && (
            <div className="flex items-center justify-between gap-3 border-b border-black/10 bg-black/[0.02] px-4 py-2.5">
              <span className="text-sm font-medium text-fg">{selectedIds.length} selected</span>
              <button type="button"
                onClick={() => { if (confirm(`Delete ${selectedIds.length} sites? Employee links will be reset.`)) bulkDel.mutate(selectedIds) }}
                className="flex h-8 items-center gap-1.5 rounded-lg bg-[#14141a] px-3 text-xs font-semibold text-white hover:opacity-90">
                <Trash2 size={13} /> Delete selected
              </button>
            </div>
          )}

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-black/10 text-xs font-semibold uppercase tracking-wide text-fg-faint">
                  {canWrite && (
                    <th className="w-10 px-4 py-3">
                      <input type="checkbox" checked={allOnPageSelected} onChange={toggleAll}
                        className="h-4 w-4 accent-[#14141a]" aria-label="Select all" />
                    </th>
                  )}
                  <th className="px-4 py-3 text-left">Site</th>
                  <th className="px-4 py-3 text-left">Office hours</th>
                  <th className="px-4 py-3 text-left">Worker hours</th>
                  <th className="px-4 py-3 text-center">Geofence</th>
                  <th className="px-4 py-3 text-right">Employees</th>
                  {canWrite && <th className="px-4 py-3 text-center">Actions</th>}
                </tr>
              </thead>
              <tbody>
                {rows.map((s) => (
                  <tr key={s.id} className="border-b border-black/[0.06] transition-colors last:border-0 hover:bg-black/[0.03]">
                    {canWrite && (
                      <td className="px-4 py-3">
                        <input type="checkbox" checked={selected.has(s.id)} onChange={() => toggle(s.id)}
                          className="h-4 w-4 accent-[#14141a]" aria-label={`Select ${s.name}`} />
                      </td>
                    )}
                    <td className="cursor-pointer px-4 py-3 font-semibold text-fg" onClick={() => navigate(`/dashboard/sites/${s.id}-v2`)}>
                      {s.name}
                    </td>
                    <td className="px-4 py-3 text-fg-muted">{schedule(s.office_start, s.office_end)}</td>
                    <td className="px-4 py-3 text-fg-muted">{schedule(s.worker_start, s.worker_end)}</td>
                    <td className="px-4 py-3 text-center">
                      {s.has_geofence ? (
                        <button type="button" onClick={() => navigate(`/dashboard/sites/${s.id}-v2`)}
                          className="mx-auto inline-flex" title="See geo location on map">
                          <Pill tone="outline"><MapPin size={11} /> set</Pill>
                        </button>
                      ) : (
                        <span className="text-fg-faint">—</span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <span className="inline-flex items-center gap-1.5 font-semibold text-fg">
                        <Users size={14} className="text-fg-faint" />
                        {s.employee_count.toLocaleString()}
                      </span>
                    </td>
                    {canWrite && (
                      <td className="px-4 py-3">
                        <div className="flex items-center justify-center gap-1">
                          <button type="button" onClick={() => setEditing(s)} title="Edit"
                            className="grid h-8 w-8 place-items-center rounded-lg text-fg-muted hover:bg-black/[0.05] hover:text-fg">
                            <Pencil size={15} />
                          </button>
                          <button type="button" title="Delete"
                            onClick={() => { if (confirm(`Delete ${s.name}? Employee links will be reset.`)) del.mutate(s.id) }}
                            className="grid h-8 w-8 place-items-center rounded-lg text-fg-muted hover:bg-black/[0.05] hover:text-fg">
                            <Trash2 size={15} />
                          </button>
                        </div>
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {pg && pg.num_pages > 1 && (
            <div className="flex items-center justify-between border-t border-black/10 px-4 py-3 text-sm text-fg-muted">
              <span className={isFetching ? 'opacity-50' : ''}>
                {pg.start_index.toLocaleString()}–{pg.end_index.toLocaleString()} of {pg.total_items.toLocaleString()}
              </span>
              <div className="flex items-center gap-1">
                <button type="button" onClick={() => setPage((p) => Math.max(1, p - 1))} disabled={!pg.has_previous}
                  className="grid h-8 w-8 place-items-center rounded-lg text-fg-muted transition-colors hover:bg-black/[0.05] hover:text-fg disabled:cursor-not-allowed disabled:opacity-40">
                  <ChevronLeft size={16} />
                </button>
                <span className="px-2 text-xs">Page {pg.current_page} / {pg.num_pages}</span>
                <button type="button" onClick={() => setPage((p) => p + 1)} disabled={!pg.has_next}
                  className="grid h-8 w-8 place-items-center rounded-lg text-fg-muted transition-colors hover:bg-black/[0.05] hover:text-fg disabled:cursor-not-allowed disabled:opacity-40">
                  <ChevronRight size={16} />
                </button>
              </div>
            </div>
          )}
        </GlassCard>
      )}

      {editing !== null && (
        <SiteFormDialog site={editing === 'new' ? null : editing} onClose={() => setEditing(null)} />
      )}
      {importing && <SiteImportDialog onClose={() => setImporting(false)} />}
    </div>
  )
}
