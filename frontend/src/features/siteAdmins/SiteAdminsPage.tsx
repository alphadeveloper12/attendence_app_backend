import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Plus, Pencil, Trash2, Shield, Eye } from 'lucide-react'
import {
  getSiteAdmins,
  createSiteAdmin,
  updateSiteAdmin,
  deleteSiteAdmins,
  type SiteAdminRow,
  type SiteAdminInput,
} from '../../lib/api/siteAdmins'
import { PageHeader } from '../../components/ui/PageHeader'
import { GlassCard } from '../../components/ui/GlassCard'
import { EmptyState } from '../../components/ui/EmptyState'
import { SkeletonRows } from '../../components/ui/Skeleton'
import { SearchInput } from '../../components/ui/Field'
import { Pill } from '../../components/ui/Pill'
import { Dialog } from '../../components/ui/Dialog'
import { useDebounce } from '../../lib/hooks/useDebounce'
import { cn } from '../../lib/utils'

const inputCls =
  'h-10 w-full rounded-xl border border-black/10 bg-white px-3 text-sm text-fg outline-none ' +
  'focus:border-black/25 focus:ring-2 focus:ring-black/10'

/**
 * Site Admins — superuser-only management of admin/viewer accounts and their
 * site assignments. Add/edit run through a modal; viewers implicitly cover all
 * sites, so the site picker is only shown for the "site admin" role.
 */
export default function SiteAdminsPage() {
  const qc = useQueryClient()
  const [search, setSearch] = useState('')
  const debounced = useDebounce(search, 300)
  const [editing, setEditing] = useState<SiteAdminRow | null>(null)
  const [creating, setCreating] = useState(false)

  const { data, isLoading } = useQuery({
    queryKey: ['site-admins', debounced],
    queryFn: () => getSiteAdmins({ search: debounced || undefined }),
  })

  const refresh = () => qc.invalidateQueries({ queryKey: ['site-admins'] })
  const remove = useMutation({ mutationFn: (id: number) => deleteSiteAdmins([id]), onSuccess: refresh })

  const admins = data?.results ?? []
  const allSites = data?.sites ?? []

  return (
    <div>
      <PageHeader
        title="Site Admins"
        subtitle={data ? `${admins.length} admin accounts` : 'Loading…'}
        actions={
          <div className="flex items-center gap-2">
            <SearchInput value={search} onChange={setSearch} placeholder="Search admins…" className="w-52" />
            <button
              type="button"
              onClick={() => setCreating(true)}
              className="flex h-10 shrink-0 items-center gap-1.5 rounded-xl bg-[#14141a] px-4 text-sm font-semibold text-white transition-opacity hover:opacity-90"
            >
              <Plus size={15} /> Add admin
            </button>
          </div>
        }
      />

      {isLoading ? (
        <GlassCard className="p-4">
          <SkeletonRows rows={6} />
        </GlassCard>
      ) : admins.length === 0 ? (
        <GlassCard>
          <EmptyState message="No admin accounts found." />
        </GlassCard>
      ) : (
        <GlassCard className="overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-black/10 text-xs font-semibold uppercase tracking-wide text-fg-faint">
                  <th className="px-4 py-3 text-left">User</th>
                  <th className="px-4 py-3 text-left">Role</th>
                  <th className="px-4 py-3 text-left">Sites</th>
                  <th className="px-4 py-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {admins.map((a) => (
                  <tr key={a.id} className="border-b border-black/[0.06] last:border-0">
                    <td className="px-4 py-3">
                      <p className="font-semibold text-fg">{a.username}</p>
                      <p className="text-xs text-fg-faint">{a.email || '—'}</p>
                    </td>
                    <td className="px-4 py-3">
                      <Pill tone={a.role === 'viewer' ? 'soft' : 'outline'}>
                        {a.role === 'viewer' ? <Eye size={11} /> : <Shield size={11} />}
                        {a.role_display}
                      </Pill>
                    </td>
                    <td className="px-4 py-3 text-fg-muted">{a.site_name}</td>
                    <td className="px-4 py-3">
                      <div className="flex items-center justify-end gap-1">
                        <button type="button" onClick={() => setEditing(a)} title="Edit"
                          className="grid h-8 w-8 place-items-center rounded-lg text-fg-muted hover:bg-black/[0.05] hover:text-fg">
                          <Pencil size={15} />
                        </button>
                        <button type="button" title="Delete"
                          onClick={() => { if (confirm(`Delete admin "${a.username}"?`)) remove.mutate(a.id) }}
                          className="grid h-8 w-8 place-items-center rounded-lg text-fg-muted hover:bg-black/[0.05] hover:text-fg">
                          <Trash2 size={15} />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </GlassCard>
      )}

      {/* Create */}
      {creating && (
        <AdminFormDialog
          open={creating}
          onClose={() => setCreating(false)}
          allSites={allSites}
          onSaved={() => { setCreating(false); refresh() }}
        />
      )}
      {/* Edit */}
      {editing && (
        <AdminFormDialog
          open={!!editing}
          onClose={() => setEditing(null)}
          allSites={allSites}
          existing={editing}
          onSaved={() => { setEditing(null); refresh() }}
        />
      )}
    </div>
  )
}

/**
 * Add/edit dialog. On create, password is required; on edit it's optional
 * (blank = keep current). The backend redirects rather than returning JSON, so
 * we treat a resolved request as success and refetch the list.
 */
function AdminFormDialog({
  open,
  onClose,
  allSites,
  existing,
  onSaved,
}: {
  open: boolean
  onClose: () => void
  allSites: { id: number; name: string }[]
  existing?: SiteAdminRow
  onSaved: () => void
}) {
  const isEdit = !!existing
  const [username, setUsername] = useState(existing?.username ?? '')
  const [email, setEmail] = useState(existing?.email ?? '')
  const [password, setPassword] = useState('')
  const [role, setRole] = useState<'site_admin' | 'viewer'>(
    existing?.role === 'viewer' ? 'viewer' : 'site_admin',
  )
  const [sites, setSites] = useState<number[]>(existing?.site_ids ?? [])

  const save = useMutation({
    mutationFn: (input: SiteAdminInput) =>
      isEdit ? updateSiteAdmin(existing!.id, input) : createSiteAdmin(input),
    onSuccess: onSaved,
  })

  const canSubmit = username.trim() && email.trim() && (isEdit || password) &&
    (role === 'viewer' || sites.length > 0)

  function toggleSite(id: number) {
    setSites((prev) => (prev.includes(id) ? prev.filter((s) => s !== id) : [...prev, id]))
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => !o && onClose()}
      title={isEdit ? `Edit ${existing!.username}` : 'Add site admin'}
      description={isEdit ? 'Leave password blank to keep the current one.' : undefined}
      footer={
        <>
          <button type="button" onClick={onClose}
            className="h-10 rounded-xl border border-black/10 px-4 text-sm font-semibold text-fg hover:bg-black/[0.04]">
            Cancel
          </button>
          <button
            type="button"
            disabled={!canSubmit || save.isPending}
            onClick={() =>
              save.mutate({ username: username.trim(), email: email.trim(), password: password || undefined, role, sites })
            }
            className="h-10 rounded-xl bg-[#14141a] px-4 text-sm font-semibold text-white hover:opacity-90 disabled:opacity-40"
          >
            {save.isPending ? 'Saving…' : 'Save'}
          </button>
        </>
      }
    >
      <div className="space-y-3">
        <Labeled label="Username">
          <input value={username} onChange={(e) => setUsername(e.target.value)} className={inputCls} />
        </Labeled>
        <Labeled label="Email">
          <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} className={inputCls} />
        </Labeled>
        <Labeled label={isEdit ? 'New password (optional)' : 'Password'}>
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} className={inputCls} />
        </Labeled>

        {/* Role toggle */}
        <Labeled label="Role">
          <div className="inline-flex rounded-xl bg-black/[0.04] p-1">
            {(['site_admin', 'viewer'] as const).map((r) => (
              <button key={r} type="button" onClick={() => setRole(r)}
                className={cn(
                  'rounded-lg px-4 py-1.5 text-sm font-semibold transition-all',
                  role === r ? 'bg-white text-fg shadow-sm' : 'text-fg-muted hover:text-fg',
                )}>
                {r === 'viewer' ? 'Viewer' : 'Site admin'}
              </button>
            ))}
          </div>
        </Labeled>

        {/* Sites (site admins only; viewers cover all sites) */}
        {role === 'site_admin' ? (
          <Labeled label={`Sites (${sites.length} selected)`}>
            <div className="max-h-44 space-y-1 overflow-y-auto rounded-xl border border-black/10 p-2">
              {allSites.map((s) => (
                <label key={s.id} className="flex cursor-pointer items-center gap-2 rounded-lg px-2 py-1.5 hover:bg-black/[0.03]">
                  <input type="checkbox" checked={sites.includes(s.id)} onChange={() => toggleSite(s.id)}
                    className="h-4 w-4 accent-[#14141a]" />
                  <span className="text-sm text-fg">{s.name}</span>
                </label>
              ))}
            </div>
          </Labeled>
        ) : (
          <p className="rounded-xl bg-black/[0.03] px-3 py-2.5 text-xs text-fg-muted">
            Viewers can see all sites (read-only).
          </p>
        )}
      </div>
    </Dialog>
  )
}

function Labeled({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-fg-faint">{label}</label>
      {children}
    </div>
  )
}
