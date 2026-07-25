import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ArrowRight, Users } from 'lucide-react'
import {
  getManpowerRecommendations, MANPOWER_PRESETS, type ManpowerSensitivity,
} from '../../lib/api/analytics'
import { useMe } from '../../lib/hooks/useMe'
import { PageHeader } from '../../components/ui/PageHeader'
import { StatTile } from '../../components/ui/StatTile'
import { GlassCard } from '../../components/ui/GlassCard'
import { EmptyState } from '../../components/ui/EmptyState'
import { SkeletonRows } from '../../components/ui/Skeleton'
import { Select } from '../../components/ui/Field'
import { Pill } from '../../components/ui/Pill'

/**
 * Manpower Recommendations — for each (department, position) the server compares
 * every site's headcount to the median and proposes moves from over-staffed to
 * under-staffed sites. Each recommendation is a card listing its moves. Tiles
 * summarise the totals; the site filter scopes to moves touching that site.
 */
const SENSITIVITY_OPTS = [
  { value: 'strict', label: 'Strict (only big gaps)' },
  { value: 'default', label: 'Default' },
  { value: 'aggressive', label: 'Aggressive (suggest more)' },
]

export default function ManpowerRecommendationsPage() {
  const { data: me } = useMe()
  const [site, setSite] = useState('')
  const [sensitivity, setSensitivity] = useState<ManpowerSensitivity>('default')
  const preset = MANPOWER_PRESETS[sensitivity]

  const { data, isLoading } = useQuery({
    queryKey: ['manpower-recommendations', site, sensitivity],
    queryFn: () =>
      getManpowerRecommendations({
        site: site || undefined,
        over_factor: preset.over,
        under_factor: preset.under,
      }),
  })

  const siteOptions = (me?.sites ?? []).map((s) => ({ value: String(s.id), label: s.name }))
  const recs = data?.recommendations ?? []
  const totalImpact = recs.reduce((sum, r) => sum + r.impact, 0)

  return (
    <div>
      <PageHeader
        title="Manpower Recommendations"
        subtitle={
          data
            ? `${data.count.toLocaleString()} rebalancing opportunities · as of ${data.as_of}`
            : 'Analysing…'
        }
        actions={
          <div className="flex flex-wrap items-center gap-2">
            <Select
              aria-label="Sensitivity"
              value={sensitivity}
              onChange={(v) => setSensitivity(v as ManpowerSensitivity)}
              options={SENSITIVITY_OPTS}
              className="w-52"
            />
            <Select
              aria-label="Filter by site"
              value={site}
              onChange={setSite}
              options={siteOptions}
              placeholder="All sites"
              className="w-48"
            />
          </div>
        }
      />

      <div className="mb-5 grid gap-3 sm:grid-cols-2">
        <StatTile label="Opportunities" value={data?.count ?? 0} icon={<Users size={18} />}
          hint="department · position groups" />
        <StatTile label="People to redeploy" value={totalImpact} icon={<ArrowRight size={18} />}
          hint="total across all moves" />
      </div>

      {isLoading ? (
        <GlassCard className="p-4">
          <SkeletonRows rows={6} />
        </GlassCard>
      ) : recs.length === 0 ? (
        <GlassCard>
          <EmptyState
            title="Nicely balanced"
            message="No site is over- or under-staffed enough to recommend a move right now."
          />
        </GlassCard>
      ) : (
        <div className="grid gap-3 lg:grid-cols-2">
          {recs.map((r) => (
            <GlassCard key={`${r.department}|${r.position}`} className="p-5">
              <div className="mb-3 flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <p className="truncate font-semibold text-fg">{r.position}</p>
                  <p className="text-xs text-fg-faint">{r.department}</p>
                </div>
                <Pill tone="outline">move {r.impact}</Pill>
              </div>

              <p className="mb-3 text-xs text-fg-muted">
                {r.total} people · median {r.median} per site
              </p>

              <ul className="space-y-2">
                {r.moves.map((m, i) => (
                  <li
                    key={i}
                    className="flex items-center gap-2 rounded-xl bg-black/[0.03] px-3 py-2 text-sm"
                  >
                    <span className="min-w-0 flex-1 truncate text-fg">
                      {m.from_site} <span className="text-fg-faint">({m.from_count})</span>
                    </span>
                    <span className="flex shrink-0 items-center gap-1 font-semibold text-fg">
                      <ArrowRight size={14} />
                      {m.quantity}
                    </span>
                    <span className="min-w-0 flex-1 truncate text-right text-fg">
                      {m.to_site} <span className="text-fg-faint">({m.to_count})</span>
                    </span>
                  </li>
                ))}
              </ul>
            </GlassCard>
          ))}
        </div>
      )}
    </div>
  )
}
