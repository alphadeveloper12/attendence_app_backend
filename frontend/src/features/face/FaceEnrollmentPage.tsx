import { useState } from 'react'
import { useQuery, keepPreviousData } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { ChevronLeft, ChevronRight, ScanFace, Check } from 'lucide-react'
import { getFaceEnrollment } from '../../lib/api/face'
import { PageHeader } from '../../components/ui/PageHeader'
import { GlassCard } from '../../components/ui/GlassCard'
import { EmptyState } from '../../components/ui/EmptyState'
import { Skeleton } from '../../components/ui/Skeleton'
import { Select, SearchInput } from '../../components/ui/Field'
import { Pill } from '../../components/ui/Pill'
import { useDebounce } from '../../lib/hooks/useDebounce'
import { cn } from '../../lib/utils'

const FACE_STATUS = [
  { value: 'all', label: 'All faces' },
  { value: 'enrolled', label: 'Enrolled' },
  { value: 'not_enrolled', label: 'Not enrolled' },
]
const EMP_STATUS = ['all', 'Active', 'Leave', 'Resigned', 'Terminated', 'No Renewal', 'Absconding'].map(
  (v) => ({ value: v, label: v === 'all' ? 'All statuses' : v }),
)

/** Initials from a name, for the avatar fallback. */
function initials(name: string) {
  return name.split(/\s+/).slice(0, 2).map((w) => w[0]).join('').toUpperCase()
}

/**
 * Face Enrollment — a photo grid of employees showing who has a face embedding
 * enrolled. Filter by site (superuser), face status, employment status and free
 * text. Cards link to the employee detail where enrollment is managed.
 */
export default function FaceEnrollmentPage() {
  const [site, setSite] = useState('all')
  const [status, setStatus] = useState('all')
  const [empStatus, setEmpStatus] = useState('all')
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(1)
  const navigate = useNavigate()
  const debounced = useDebounce(search, 300)

  const { data, isLoading, isFetching } = useQuery({
    queryKey: ['face', site, status, empStatus, debounced, page],
    queryFn: () =>
      getFaceEnrollment({
        site,
        status,
        emp_status: empStatus,
        search: debounced || undefined,
        page,
        per_page: 20,
      }),
    placeholderData: keepPreviousData,
  })

  const employees = data?.employees ?? []
  const pg = data?.pagination
  const enrolled = data?.employees.filter((e) => e.face_enrolled).length ?? 0

  const reset = () => setPage(1)

  return (
    <div>
      <PageHeader
        title="Face Enrollment"
        subtitle={
          data
            ? `${data.total_count.toLocaleString()} employees · ${enrolled} enrolled on this page`
            : 'Loading…'
        }
        actions={
          <div className="flex flex-wrap items-center gap-2">
            {data?.permissions.is_superuser && data.sites.length > 0 && (
              <Select
                aria-label="Filter by site"
                value={site}
                onChange={(v) => { setSite(v); reset() }}
                options={[{ value: 'all', label: 'All sites' }, ...data.sites.map((s) => ({ value: String(s.id), label: s.name }))]}
                className="w-40"
              />
            )}
            <Select aria-label="Face status" value={status}
              onChange={(v) => { setStatus(v); reset() }} options={FACE_STATUS} className="w-40" />
            <Select aria-label="Employment status" value={empStatus}
              onChange={(v) => { setEmpStatus(v); reset() }} options={EMP_STATUS} className="w-40" />
            <SearchInput value={search} onChange={(v) => { setSearch(v); reset() }}
              placeholder="Name or badge…" className="w-48" />
          </div>
        }
      />

      {isLoading ? (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
          {Array.from({ length: 15 }).map((_, i) => (
            <Skeleton key={i} className="h-44 rounded-2xl" />
          ))}
        </div>
      ) : employees.length === 0 ? (
        <GlassCard>
          <EmptyState message="No employees match these filters." />
        </GlassCard>
      ) : (
        <div className={cn('grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5', isFetching && 'opacity-60')}>
          {employees.map((e) => (
            <button
              key={e.id}
              type="button"
              onClick={() => navigate(`/dashboard/user/${e.id}-v2`)}
              className="glass glass-hover flex flex-col items-center rounded-2xl p-4 text-center"
            >
              <div className="relative mb-3">
                {e.profile_picture ? (
                  <img
                    src={e.profile_picture}
                    alt={e.name}
                    className="h-16 w-16 rounded-full object-cover"
                  />
                ) : (
                  <span className="grid h-16 w-16 place-items-center rounded-full bg-black/[0.06] text-lg font-bold text-fg-muted">
                    {initials(e.name)}
                  </span>
                )}
                <span
                  className={cn(
                    'absolute -bottom-1 -right-1 grid h-6 w-6 place-items-center rounded-full border-2 border-white',
                    e.face_enrolled ? 'bg-[#14141a] text-white' : 'bg-black/[0.08] text-fg-faint',
                  )}
                  title={e.face_enrolled ? 'Face enrolled' : 'Not enrolled'}
                >
                  {e.face_enrolled ? <Check size={13} /> : <ScanFace size={13} />}
                </span>
              </div>
              <p className="line-clamp-1 text-sm font-semibold text-fg">{e.name}</p>
              <p className="text-xs text-fg-faint">#{e.badge_number}</p>
              <p className="mt-0.5 line-clamp-1 text-xs text-fg-muted">{e.site_name}</p>
              <span className="mt-2">
                <Pill tone={e.face_enrolled ? 'solid' : 'soft'}>
                  {e.face_enrolled ? 'Enrolled' : 'Not enrolled'}
                </Pill>
              </span>
            </button>
          ))}
        </div>
      )}

      {pg && pg.total_pages > 1 && (
        <div className="mt-4 flex items-center justify-center gap-3 text-sm text-fg-muted">
          <button type="button" onClick={() => setPage((p) => Math.max(1, p - 1))} disabled={!pg.has_previous}
            className="grid h-8 w-8 place-items-center rounded-lg hover:bg-black/[0.05] hover:text-fg disabled:opacity-40">
            <ChevronLeft size={16} />
          </button>
          <span className="text-xs">Page {pg.current_page} / {pg.total_pages}</span>
          <button type="button" onClick={() => setPage((p) => p + 1)} disabled={!pg.has_next}
            className="grid h-8 w-8 place-items-center rounded-lg hover:bg-black/[0.05] hover:text-fg disabled:opacity-40">
            <ChevronRight size={16} />
          </button>
        </div>
      )}
    </div>
  )
}
