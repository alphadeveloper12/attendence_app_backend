import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { Users, UserCheck, Clock, UserX, Building2, Plus, Download, Upload, FileEdit } from 'lucide-react'
import { getStats } from '../../lib/api/dashboard'
import { exportEmployeesUrl, type EmployeeListParams } from '../../lib/api/employees'
import { useMe } from '../../lib/hooks/useMe'
import { PageHeader } from '../../components/ui/PageHeader'
import { StatTile } from '../../components/ui/StatTile'
import { Select, SearchInput } from '../../components/ui/Field'
import { useDebounce } from '../../lib/hooks/useDebounce'
import { EmployeeTable } from './EmployeeTable'
import { ExecSummaryCard } from './ExecSummaryCard'
import { AlertsPanel } from './AlertsPanel'
import { SiteActivityPanel } from './SiteActivityPanel'
import { ImportWizard } from './ImportWizard'
import { BulkEditWizard } from './BulkEditWizard'

const opt = (values: string[] = [], allLabel: string) => [
  { value: 'all', label: allLabel },
  ...values.map((v) => ({ value: v, label: v })),
]

/**
 * Main dashboard — KPI tiles (present/late/absent double as table filters), a
 * filter bar, the workforce table with bulk actions + add/edit, plus the
 * executive summary and live geofence alerts. Stats poll every 60s.
 */
export default function DashboardPage() {
  const { data: me } = useMe()
  const navigate = useNavigate()
  const canWrite = !!me?.can_write
  const isSuper = !!me?.is_superuser

  // Filters shared between stats + employee table.
  const [site, setSite] = useState('all')
  const [status, setStatus] = useState('all')
  const [category, setCategory] = useState('all')
  const [employer, setEmployer] = useState('all')
  const [search, setSearch] = useState('')
  // KPI-card attendance filter: 'total' (all) | 'present' | 'late' | 'absent'.
  const [attendance, setAttendance] = useState('total')
  const debounced = useDebounce(search, 300)

  const [importing, setImporting] = useState(false)
  const [bulkEditing, setBulkEditing] = useState(false)

  const statsParams = { site, status, category, employer }
  const { data: stats } = useQuery({
    queryKey: ['stats', statsParams],
    queryFn: () => getStats(statsParams),
    refetchInterval: 60_000,
  })

  const tableFilters: EmployeeListParams = useMemo(
    () => ({
      site,
      status,
      category,
      employer,
      search: debounced || undefined,
      // 'total' means no attendance filtering (all rows).
      attendance_filter: attendance !== 'total' ? attendance : undefined,
    }),
    [site, status, category, employer, debounced, attendance],
  )

  // Click a KPI card to filter the table; clicking the active card returns to 'total'.
  const setAttendanceFilter = (v: string) => setAttendance((cur) => (cur === v ? 'total' : v))

  return (
    <div>
      <PageHeader
        title="Dashboard"
        subtitle="Live workforce overview"
        actions={
          <div className="flex flex-wrap items-center gap-2">
            <a
              href={exportEmployeesUrl(tableFilters)}
              className="flex h-10 items-center gap-1.5 rounded-xl border border-black/10 bg-white/60 px-3.5 text-sm font-semibold text-fg hover:bg-white"
            >
              <Download size={15} /> Export
            </a>
            {isSuper && (
              <button
                type="button"
                onClick={() => setImporting(true)}
                className="flex h-10 items-center gap-1.5 rounded-xl border border-black/10 bg-white/60 px-3.5 text-sm font-semibold text-fg hover:bg-white"
              >
                <Upload size={15} /> Import
              </button>
            )}
            {canWrite && (
              <button
                type="button"
                onClick={() => setBulkEditing(true)}
                className="flex h-10 items-center gap-1.5 rounded-xl border border-black/10 bg-white/60 px-3.5 text-sm font-semibold text-fg hover:bg-white"
              >
                <FileEdit size={15} /> Bulk edit
              </button>
            )}
            {canWrite && (
              <button
                type="button"
                onClick={() => navigate('/dashboard/employee-edit/new-v2')}
                className="flex h-10 items-center gap-1.5 rounded-xl bg-[#14141a] px-4 text-sm font-semibold text-white hover:opacity-90"
              >
                <Plus size={15} /> Add employee
              </button>
            )}
          </div>
        }
      />

      {/* KPI tiles */}
      <div className="mb-5 grid grid-cols-2 gap-3 lg:grid-cols-5">
        <StatTile label="Employees" value={(stats?.total_employees ?? 0).toLocaleString()} icon={<Users size={18} />}
          active={attendance === 'total'} onClick={() => setAttendanceFilter('total')} />
        <StatTile label="Present today" value={stats?.today_attendance_count ?? 0} icon={<UserCheck size={18} />}
          active={attendance === 'present'} onClick={() => setAttendanceFilter('present')} />
        <StatTile label="Late" value={stats?.late_count ?? 0} icon={<Clock size={18} />}
          active={attendance === 'late'} onClick={() => setAttendanceFilter('late')} />
        <StatTile label="Absent" value={stats?.absent_today_count ?? 0} icon={<UserX size={18} />}
          active={attendance === 'absent'} onClick={() => setAttendanceFilter('absent')} />
        <StatTile label="Sites" value={stats?.total_sites ?? 0} icon={<Building2 size={18} />} />
      </div>

      {/* Filter bar */}
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <Select aria-label="Site" value={site} onChange={setSite}
          options={[{ value: 'all', label: 'All sites' }, ...(stats?.sites ?? []).map((s) => ({ value: String(s.id), label: s.name }))]}
          className="w-40" />
        <Select aria-label="Status" value={status} onChange={setStatus} options={opt(stats?.statuses, 'All statuses')} className="w-40" />
        <Select aria-label="Category" value={category} onChange={setCategory} options={opt(stats?.categories, 'All categories')} className="w-40" />
        <Select aria-label="Employer" value={employer} onChange={setEmployer} options={opt(stats?.employers, 'All employers')} className="w-40" />
        <SearchInput value={search} onChange={setSearch} placeholder="Name or badge…" className="w-52" />
      </div>

      {/* Main grid: table + side column */}
      <div className="grid gap-4 xl:grid-cols-3">
        <div className="xl:col-span-2">
          <EmployeeTable filters={tableFilters} canWrite={canWrite}
            onEdit={(id) => navigate(`/dashboard/employee-edit/${id}-v2`)} />
        </div>
        <div className="space-y-4">
          <ExecSummaryCard />
          <AlertsPanel site={site !== 'all' ? site : undefined} />
        </div>
      </div>

      {/* Per-site activity health */}
      <div className="mt-4">
        <SiteActivityPanel onSelectSite={(id) => setSite(String(id))} />
      </div>

      {importing && <ImportWizard onClose={() => setImporting(false)} />}
      {bulkEditing && <BulkEditWizard onClose={() => setBulkEditing(false)} />}
    </div>
  )
}
