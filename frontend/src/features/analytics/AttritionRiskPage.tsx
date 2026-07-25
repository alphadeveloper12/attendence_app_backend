import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import type { ColumnDef } from '@tanstack/react-table'
import { useNavigate } from 'react-router-dom'
import { getAttritionRisk, type AttritionRow } from '../../lib/api/analytics'
import { useMe } from '../../lib/hooks/useMe'
import { PageHeader } from '../../components/ui/PageHeader'
import { StatTile } from '../../components/ui/StatTile'
import { DataTable } from '../../components/ui/DataTable'
import { Select } from '../../components/ui/Field'
import { Pill } from '../../components/ui/Pill'

type Band = 'high' | 'medium' | 'low'

/** Monochrome pill tone per risk band (no colour in this theme). */
const bandTone = { high: 'solid', medium: 'outline', low: 'soft' } as const

/**
 * Attrition Risk — deterministic 0–100 risk score per employee with the driving
 * factors. Band tiles filter the table; a site filter scopes the query.
 * Clicking a row opens that employee's detail page.
 */
export default function AttritionRiskPage() {
  const { data: me } = useMe()
  const [site, setSite] = useState('')
  const [band, setBand] = useState<Band | ''>('')
  const navigate = useNavigate()

  const { data, isLoading } = useQuery({
    queryKey: ['attrition-risk', site],
    queryFn: () => getAttritionRisk({ site: site || undefined }),
  })

  // Band tiles filter client-side for instant feedback.
  const rows = useMemo(
    () => (data?.rows ?? []).filter((r) => !band || r.band === band),
    [data, band],
  )

  const columns = useMemo<ColumnDef<AttritionRow, any>[]>(
    () => [
      { accessorKey: 'name', header: 'Employee', cell: (c) => (
        <div>
          <p className="font-semibold text-fg">{c.getValue<string>()}</p>
          <p className="text-xs text-fg-faint">#{c.row.original.badge_number}</p>
        </div>
      ) },
      { accessorKey: 'site', header: 'Site' },
      { accessorKey: 'department', header: 'Department' },
      { accessorKey: 'position', header: 'Position' },
      {
        accessorKey: 'score',
        header: 'Score',
        cell: (c) => (
          <Pill tone={bandTone[c.row.original.band]}>{c.getValue<number>()}</Pill>
        ),
      },
      {
        id: 'factors',
        header: 'Top factor',
        enableSorting: false,
        cell: (c) => (
          <span className="text-sm text-fg-muted">
            {c.row.original.factors[0]?.label ?? '—'}
          </span>
        ),
      },
    ],
    [],
  )

  const siteOptions = (me?.sites ?? []).map((s) => ({ value: s.name, label: s.name }))
  const bands = data?.bands

  return (
    <div>
      <PageHeader
        title="Attrition Risk"
        subtitle={
          data ? `${data.count.toLocaleString()} employees scored · as of ${data.as_of}` : 'Scoring…'
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

      {/* Band tiles — click to filter */}
      <div className="mb-5 grid gap-3 sm:grid-cols-3">
        <StatTile label="High risk" value={bands?.high ?? 0} active={band === 'high'}
          onClick={() => setBand(band === 'high' ? '' : 'high')} hint="score ≥ 60" />
        <StatTile label="Medium risk" value={bands?.medium ?? 0} active={band === 'medium'}
          onClick={() => setBand(band === 'medium' ? '' : 'medium')} hint="score 35–59" />
        <StatTile label="Low risk" value={bands?.low ?? 0} active={band === 'low'}
          onClick={() => setBand(band === 'low' ? '' : 'low')} hint="score < 35" />
      </div>

      <DataTable
        columns={columns}
        data={rows}
        loading={isLoading}
        pageSize={15}
        onRowClick={(r) => navigate(`/dashboard/user/${r.employee_id}-v2`)}
        emptyMessage="No employees match the current filters."
      />
    </div>
  )
}
