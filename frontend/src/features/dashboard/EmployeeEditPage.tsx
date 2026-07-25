import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate, useParams } from 'react-router-dom'
import { ArrowLeft, Save } from 'lucide-react'
import { useMe } from '../../lib/hooks/useMe'
import { getSites } from '../../lib/api/sites'
import { useEmployeeForm } from './useEmployeeForm'
import { GlassCard } from '../../components/ui/GlassCard'
import { Skeleton } from '../../components/ui/Skeleton'
import { cn } from '../../lib/utils'

/** Strip the `-v2` suffix; `new` means add mode, anything else is an employee id. */
function useEditTarget(): { isNew: boolean; id: number | null } {
  const { idv2 = '' } = useParams()
  const raw = idv2.replace(/-v2$/, '')
  if (raw === 'new') return { isNew: true, id: null }
  return { isNew: false, id: Number(raw) }
}

/**
 * Employee editor as a full page (replaces the old modal). A vertical tab
 * sidebar on the left switches sections; the active section's fields fill the
 * panel on the right. A back button returns to the dashboard. Add mode uses the
 * `new` target; edit mode prefills from the server.
 */
export default function EmployeeEditPage() {
  const { id, isNew } = useEditTarget()
  const navigate = useNavigate()
  const { data: me } = useMe()
  const back = () => navigate('/dashboard-v2')

  // Full (server-scoped) site list so the Site dropdown shows names, not ids —
  // superusers have an empty `me.sites`, so we can't rely on it alone.
  const { data: sitesData } = useQuery({
    queryKey: ['sites-all'],
    queryFn: () => getSites({ per_page: 1000 }),
  })
  const sites = (sitesData?.results ?? []).map((s) => ({ id: s.id, name: s.name }))

  const { tabs, save, isLoading, error, setError, isEdit, name } = useEmployeeForm({
    employeeId: id,
    isSuperuser: !!me?.is_superuser,
    sites: sites.length ? sites : (me?.sites ?? []),
    onSaved: back,
  })

  const [active, setActive] = useState('employment')
  const activeTab = tabs.find((t) => t.value === active) ?? tabs[0]
  const loadingPrefill = isEdit && isLoading

  return (
    <div>
      {/* Header */}
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div className="min-w-0">
          <button type="button" onClick={back}
            className="mb-2 inline-flex items-center gap-1.5 text-sm font-medium text-fg-muted transition-colors hover:text-fg">
            <ArrowLeft size={15} /> Back to dashboard
          </button>
          <h1 className="truncate text-2xl font-bold text-fg">{isNew ? 'Add employee' : name || 'Edit employee'}</h1>
          <p className="text-sm text-fg-muted">{isNew ? 'Create a new employee record' : 'Update this employee’s details'}</p>
        </div>
        <div className="flex items-center gap-2">
          <button type="button" onClick={back}
            className="h-10 rounded-xl border border-black/10 bg-white/60 px-4 text-sm font-semibold text-fg hover:bg-white">
            Cancel
          </button>
          <button type="button" disabled={save.isPending || loadingPrefill}
            onClick={() => { setError(null); save.mutate() }}
            className="flex h-10 items-center gap-1.5 rounded-xl bg-[#14141a] px-5 text-sm font-semibold text-white hover:opacity-90 disabled:opacity-40">
            <Save size={15} /> {save.isPending ? 'Saving…' : isEdit ? 'Save changes' : 'Create employee'}
          </button>
        </div>
      </div>

      {error && (
        <p className="mb-4 rounded-xl bg-black/[0.04] px-3 py-2 text-sm font-medium text-fg">{error}</p>
      )}

      {loadingPrefill ? (
        <div className="grid gap-4 lg:grid-cols-[220px_1fr]">
          <Skeleton className="h-72 rounded-3xl" />
          <Skeleton className="h-72 rounded-3xl" />
        </div>
      ) : (
        <div className="grid gap-4 lg:grid-cols-[220px_1fr]">
          {/* Tab sidebar */}
          <GlassCard className="h-fit p-2">
            <nav className="flex gap-1 overflow-x-auto lg:flex-col">
              {tabs.map((t) => (
                <button key={t.value} type="button" onClick={() => setActive(t.value)}
                  className={cn(
                    'shrink-0 rounded-xl px-3.5 py-2.5 text-left text-sm font-semibold transition-all lg:w-full',
                    active === t.value ? 'bg-[#14141a] text-white' : 'text-fg-muted hover:bg-black/[0.04] hover:text-fg',
                  )}>
                  {t.label}
                </button>
              ))}
            </nav>
          </GlassCard>

          {/* Edit panel */}
          <GlassCard className="p-5">
            <h2 className="mb-4 text-sm font-bold uppercase tracking-wide text-fg-muted">{activeTab.label}</h2>
            {activeTab.content}
            {isNew && (
              <p className="mt-4 text-xs text-fg-faint">
                Insurance, attachments, sick leave and history become available after the employee is created.
              </p>
            )}
          </GlassCard>
        </div>
      )}
    </div>
  )
}
