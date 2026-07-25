import type { ReactNode } from 'react'
import {
  ResponsiveContainer,
  BarChart as ReBarChart,
  Bar,
  LineChart as ReLineChart,
  Line,
  PieChart as RePieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from 'recharts'

/**
 * Recharts wrappers styled for the light-glass monochrome theme. All charts are
 * black-and-white (dark ink on light), sharing one greyscale palette so the
 * dashboard, reports and monthly pages look consistent. Each is responsive and
 * renders inside whatever card the caller provides.
 */

/** Greyscale series palette (darkest first) — used for multi-slice charts. */
export const CHART_PALETTE = ['#14141a', '#52525b', '#71717a', '#a1a1aa', '#d4d4d8', '#e4e4e7']

const AXIS = { stroke: '#a1a1aa', fontSize: 12, tickLine: false, axisLine: false }
const GRID = '#00000010'

/** Shared tooltip chrome — a small glass card. */
const tooltipStyle = {
  borderRadius: 12,
  border: '1px solid rgba(17,17,26,0.08)',
  background: 'rgba(255,255,255,0.92)',
  backdropFilter: 'blur(8px)',
  boxShadow: '0 10px 30px -12px rgba(17,17,26,0.25)',
  fontSize: 12,
  color: '#14141a',
}

/** Vertical bar chart for a single series (e.g. hourly check-ins). */
export function BarChart({
  data,
  xKey,
  yKey,
  height = 260,
  color = '#14141a',
}: {
  data: Record<string, any>[]
  xKey: string
  yKey: string
  height?: number
  color?: string
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ReBarChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: -12 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={GRID} vertical={false} />
        <XAxis dataKey={xKey} {...AXIS} />
        <YAxis {...AXIS} allowDecimals={false} width={40} />
        <Tooltip contentStyle={tooltipStyle} cursor={{ fill: '#00000008' }} />
        <Bar dataKey={yKey} fill={color} radius={[6, 6, 0, 0]} maxBarSize={44} />
      </ReBarChart>
    </ResponsiveContainer>
  )
}

/** Line/trend chart for one or more series. */
export function LineChart({
  data,
  xKey,
  series,
  height = 260,
}: {
  data: Record<string, any>[]
  xKey: string
  series: { key: string; label?: string; color?: string }[]
  height?: number
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ReLineChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: -12 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={GRID} vertical={false} />
        <XAxis dataKey={xKey} {...AXIS} />
        <YAxis {...AXIS} allowDecimals={false} width={40} />
        <Tooltip contentStyle={tooltipStyle} />
        {series.map((s, i) => (
          <Line
            key={s.key}
            type="monotone"
            dataKey={s.key}
            name={s.label ?? s.key}
            stroke={s.color ?? CHART_PALETTE[i % CHART_PALETTE.length]}
            strokeWidth={2.5}
            dot={false}
            activeDot={{ r: 4 }}
          />
        ))}
      </ReLineChart>
    </ResponsiveContainer>
  )
}

/** Doughnut/pie chart with a greyscale palette + a centre label. */
export function DoughnutChart({
  data,
  nameKey,
  valueKey,
  height = 260,
  centerLabel,
}: {
  data: Record<string, any>[]
  nameKey: string
  valueKey: string
  height?: number
  centerLabel?: ReactNode
}) {
  return (
    <div className="relative" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <RePieChart>
          <Tooltip contentStyle={tooltipStyle} />
          <Pie
            data={data}
            dataKey={valueKey}
            nameKey={nameKey}
            innerRadius="58%"
            outerRadius="82%"
            paddingAngle={2}
            stroke="none"
          >
            {data.map((_, i) => (
              <Cell key={i} fill={CHART_PALETTE[i % CHART_PALETTE.length]} />
            ))}
          </Pie>
        </RePieChart>
      </ResponsiveContainer>
      {centerLabel != null && (
        <div className="pointer-events-none absolute inset-0 grid place-items-center">
          <div className="text-center">{centerLabel}</div>
        </div>
      )}
    </div>
  )
}

/**
 * A simple legend for the doughnut/pie — greyscale swatches matching the
 * palette order. Render beside a {@link DoughnutChart} with the same data.
 */
export function ChartLegend({
  items,
}: {
  items: { label: string; value?: ReactNode }[]
}) {
  return (
    <ul className="space-y-1.5">
      {items.map((it, i) => (
        <li key={it.label} className="flex items-center gap-2 text-sm">
          <span
            className="h-2.5 w-2.5 shrink-0 rounded-full"
            style={{ background: CHART_PALETTE[i % CHART_PALETTE.length] }}
          />
          <span className="flex-1 truncate text-fg-muted">{it.label}</span>
          {it.value != null && <span className="font-semibold text-fg">{it.value}</span>}
        </li>
      ))}
    </ul>
  )
}
