import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Plus, Pencil, Trash2, Check, X } from 'lucide-react'
import {
  getDepartments,
  createDepartment,
  updateDepartment,
  deleteDepartment,
  type Department,
} from '../../lib/api/departments'
import { useMe } from '../../lib/hooks/useMe'
import { PageHeader } from '../../components/ui/PageHeader'
import { GlassCard } from '../../components/ui/GlassCard'
import { EmptyState } from '../../components/ui/EmptyState'
import { SkeletonRows } from '../../components/ui/Skeleton'
import { Pill } from '../../components/ui/Pill'

const inputCls =
  'h-9 w-full rounded-lg border border-black/10 bg-white px-3 text-sm text-fg outline-none ' +
  'focus:border-black/25 focus:ring-2 focus:ring-black/10'

/**
 * Departments Management — the department catalogue that seeds dropdowns across
 * the app. Superusers can add, rename, reassign managers and (soft-)delete.
 * Editing is inline; deletes hide the department but keep employee history.
 */
export default function DepartmentsPage() {
  const { data: me } = useMe()
  const qc = useQueryClient()
  const canWrite = !!me?.can_write

  const { data, isLoading } = useQuery({ queryKey: ['departments'], queryFn: getDepartments })

  // Invalidate the list after any successful write.
  const refresh = () => qc.invalidateQueries({ queryKey: ['departments'] })

  const create = useMutation({ mutationFn: createDepartment, onSuccess: refresh })
  const update = useMutation({
    mutationFn: (v: { id: number; name: string; manager_name: string }) =>
      updateDepartment(v.id, { name: v.name, manager_name: v.manager_name }),
    onSuccess: refresh,
  })
  const remove = useMutation({ mutationFn: deleteDepartment, onSuccess: refresh })

  // New-department form state.
  const [newName, setNewName] = useState('')
  const [newManager, setNewManager] = useState('')
  // Which row is being edited, plus its draft values.
  const [editing, setEditing] = useState<number | null>(null)
  const [draft, setDraft] = useState({ name: '', manager_name: '' })

  function submitNew(e: React.FormEvent) {
    e.preventDefault()
    if (!newName.trim()) return
    create.mutate(
      { name: newName.trim(), manager_name: newManager.trim() || undefined },
      { onSuccess: () => { setNewName(''); setNewManager('') } },
    )
  }

  function startEdit(d: Department) {
    setEditing(d.id)
    setDraft({ name: d.name, manager_name: d.manager_name })
  }

  function saveEdit(id: number) {
    if (!draft.name.trim()) return
    update.mutate(
      { id, name: draft.name.trim(), manager_name: draft.manager_name.trim() },
      { onSuccess: () => setEditing(null) },
    )
  }

  const departments = data?.departments ?? []

  return (
    <div>
      <PageHeader
        title="Departments"
        subtitle={data ? `${data.count} active departments` : 'Loading…'}
      />

      {/* Add form (superuser / writers only) */}
      {canWrite && (
        <GlassCard className="mb-4 p-4">
          <form onSubmit={submitNew} className="flex flex-col gap-2 sm:flex-row sm:items-center">
            <input
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="Department name"
              className={inputCls + ' sm:flex-1'}
            />
            <input
              value={newManager}
              onChange={(e) => setNewManager(e.target.value)}
              placeholder="Manager (optional)"
              className={inputCls + ' sm:flex-1'}
            />
            <button
              type="submit"
              disabled={create.isPending || !newName.trim()}
              className="flex h-9 shrink-0 items-center justify-center gap-1.5 rounded-lg bg-[#14141a] px-4 text-sm font-semibold text-white transition-opacity hover:opacity-90 disabled:opacity-40"
            >
              <Plus size={15} />
              Add
            </button>
          </form>
          {create.isError && (
            <p className="mt-2 text-xs font-medium text-fg">{(create.error as Error).message}</p>
          )}
        </GlassCard>
      )}

      {isLoading ? (
        <GlassCard className="p-4">
          <SkeletonRows rows={8} />
        </GlassCard>
      ) : departments.length === 0 ? (
        <GlassCard>
          <EmptyState message="No departments yet." />
        </GlassCard>
      ) : (
        <GlassCard className="overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-black/10 text-xs font-semibold uppercase tracking-wide text-fg-faint">
                <th className="px-4 py-3 text-left">Department</th>
                <th className="px-4 py-3 text-left">Manager</th>
                <th className="px-4 py-3 text-center">Positions</th>
                {canWrite && <th className="px-4 py-3 text-right">Actions</th>}
              </tr>
            </thead>
            <tbody>
              {departments.map((d) => {
                const isEditing = editing === d.id
                return (
                  <tr key={d.id} className="border-b border-black/[0.06] last:border-0">
                    <td className="px-4 py-2.5">
                      {isEditing ? (
                        <input
                          value={draft.name}
                          onChange={(e) => setDraft((s) => ({ ...s, name: e.target.value }))}
                          className={inputCls}
                          autoFocus
                        />
                      ) : (
                        <span className="font-semibold text-fg">{d.name}</span>
                      )}
                    </td>
                    <td className="px-4 py-2.5 text-fg-muted">
                      {isEditing ? (
                        <input
                          value={draft.manager_name}
                          onChange={(e) =>
                            setDraft((s) => ({ ...s, manager_name: e.target.value }))
                          }
                          className={inputCls}
                          placeholder="—"
                        />
                      ) : (
                        d.manager_name || '—'
                      )}
                    </td>
                    <td className="px-4 py-2.5 text-center">
                      <Pill tone="soft">{d.positions_count}</Pill>
                    </td>
                    {canWrite && (
                      <td className="px-4 py-2.5">
                        <div className="flex items-center justify-end gap-1">
                          {isEditing ? (
                            <>
                              <IconBtn onClick={() => saveEdit(d.id)} title="Save">
                                <Check size={16} />
                              </IconBtn>
                              <IconBtn onClick={() => setEditing(null)} title="Cancel">
                                <X size={16} />
                              </IconBtn>
                            </>
                          ) : (
                            <>
                              <IconBtn onClick={() => startEdit(d)} title="Edit">
                                <Pencil size={15} />
                              </IconBtn>
                              <IconBtn
                                onClick={() => {
                                  if (confirm(`Delete department "${d.name}"?`)) remove.mutate(d.id)
                                }}
                                title="Delete"
                              >
                                <Trash2 size={15} />
                              </IconBtn>
                            </>
                          )}
                        </div>
                      </td>
                    )}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </GlassCard>
      )}
    </div>
  )
}

/** Small square icon button used for the row actions. */
function IconBtn({
  children,
  onClick,
  title,
}: {
  children: React.ReactNode
  onClick: () => void
  title: string
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      className="grid h-8 w-8 place-items-center rounded-lg text-fg-muted transition-colors hover:bg-black/[0.05] hover:text-fg"
    >
      {children}
    </button>
  )
}
