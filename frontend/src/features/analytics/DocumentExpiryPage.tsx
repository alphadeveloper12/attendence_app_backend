import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import type { ColumnDef } from '@tanstack/react-table'
import { useNavigate } from 'react-router-dom'
import { getDocumentExpiry, type DocumentExpiryRow, type ExpiryDoc } from '../../lib/api/analytics'
import { useMe } from '../../lib/hooks/useMe'
import { PageHeader } from '../../components/ui/PageHeader'
import { StatTile } from '../../components/ui/StatTile'
import { DataTable } from '../../components/ui/DataTable'
import { Select } from '../../components/ui/Field'
import { Pill } from '../../components/ui/Pill'

type Bucket = 'expired' | 'critical' | 'urgent' | 'soon'

/** Monochrome pill tone per urgency bucket (theme is black & white). */
const bucketTone: Record<ExpiryDoc['bucket'], 'solid' | 'outline' | 'soft'> = {
  expired: 'solid',
  critical: 'solid',
  urgent: 'outline',
  soon: 'soft',
  ok: 'soft',
}

/** Human label for a document's remaining days. */
function daysLabel(days: number): string {
  if (days < 0) return `${Math.abs(days)}d ago`
  if (days === 0) return 'today'
  return `in ${days}d`
}

/**
 * Document Expiry — Active employees whose passport or visa is expiring soon or
 * already expired, bucketed by urgency. Bucket tiles filter the table; the site
 * filter re-queries the server. Clicking a row opens the employee.
 */
export default function DocumentExpiryPage() {
  const { data: me } = useMe()
  const [site, setSite] = useState('')
  const [bucket, setBucket] = useState<Bucket | ''>('')
  const navigate = useNavigate()

  const { data, isLoading } = useQuery({
    queryKey: ['document-expiry', site],
    queryFn: () => getDocumentExpiry({ site: site || undefined }),
  })

  // Bucket tiles filter client-side for instant feedback.
  const rows = useMemo(
    () => (data?.rows ?? []).filter((r) => !bucket || r.worst_bucket === bucket),
    [data, bucket],
  )

  const columns = useMemo<ColumnDef<DocumentExpiryRow, any>[]>(
    () => [
      {
        accessorKey: 'name',
        header: 'Employee',
        cell: (c) => (
          <div>
            <p className="font-semibold text-fg">{c.getValue<string>()}</p>
            <p className="text-xs text-fg-faint">#{c.row.original.badge_number}</p>
          </div>
        ),
      },
      { accessorKey: 'site', header: 'Site' },
      { accessorKey: 'department', header: 'Department' },
      {
        id: 'docs',
        header: 'Documents',
        enableSorting: false,
        cell: (c) => (
          <div className="flex flex-wrap gap-1.5">
            {c.row.original.docs
              .filter((d) => d.bucket !== 'ok')
              .map((d) => (
                <Pill key={d.type} tone={bucketTone[d.bucket]}>
                  {d.type} · {daysLabel(d.days)}
                </Pill>
              ))}
          </div>
        ),
      },
      {
        id: 'soonest',
        header: 'Soonest',
        accessorFn: (r) => Math.min(...r.docs.map((d) => d.days)),
        cell: (c) => <span className="text-sm text-fg-muted">{daysLabel(c.getValue<number>())}</span>,
      },
    ],
    [],
  )

  const siteOptions = (me?.sites ?? []).map((s) => ({ value: String(s.id), label: s.name }))
  const buckets = data?.buckets

  return (
    <div>
      <PageHeader
        title="Document Expiry"
        subtitle={
          data
            ? `${data.count.toLocaleString()} employees with expiring documents · as of ${data.as_of}`
            : 'Scanning…'
        }
        actions={
          <Select
            aria-label="Filter by site"
            value={site}
            onChange={setSite}
            options={siteOptions}
            placeholder="All sites"
            className="w-48"
          />
        }
      />

      {/* Bucket tiles — click to filter */}
      <div className="mb-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
        <StatTile label="All" value={data?.count ?? 0} active={bucket === ''}
          onClick={() => setBucket('')} hint="expiring soon" />
        <StatTile label="Expired" value={buckets?.expired ?? 0} active={bucket === 'expired'}
          onClick={() => setBucket(bucket === 'expired' ? '' : 'expired')} hint="past due" />
        <StatTile label="Critical" value={buckets?.critical ?? 0} active={bucket === 'critical'}
          onClick={() => setBucket(bucket === 'critical' ? '' : 'critical')} hint="≤ 14 days" />
        <StatTile label="Urgent" value={buckets?.urgent ?? 0} active={bucket === 'urgent'}
          onClick={() => setBucket(bucket === 'urgent' ? '' : 'urgent')} hint="15–30 days" />
        <StatTile label="Soon" value={buckets?.soon ?? 0} active={bucket === 'soon'}
          onClick={() => setBucket(bucket === 'soon' ? '' : 'soon')} hint="31–90 days" />
      </div>

      <DataTable
        columns={columns}
        data={rows}
        loading={isLoading}
        pageSize={15}
        onRowClick={(r) => navigate(`/dashboard/user/${r.employee_id}-v2`)}
        emptyMessage="No employees have documents expiring in this window."
      />
    </div>
  )
}
