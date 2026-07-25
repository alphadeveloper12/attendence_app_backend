import { useMemo } from 'react'
import { cn } from '../../lib/utils'

/** One day's attendance status. Keys are normalized lowercase from the API. */
export type DayStatus =
  | 'present'
  | 'late'
  | 'absent'
  | 'leave'
  | 'off'
  | 'holiday'
  | 'half'
  | string

export interface CalendarDay {
  /** ISO date, "YYYY-MM-DD". */
  date: string
  status: DayStatus
  /** Optional label shown on hover (e.g. check-in time). */
  title?: string
}

/**
 * Monochrome day-cell treatment per status. The theme is black & white, so
 * status is encoded by fill intensity + border rather than colour:
 *   present = solid dark · late = mid grey · half = light · absent = hollow
 *   leave/off/holiday = faint.
 */
const STATUS_STYLE: Record<string, string> = {
  present: 'bg-[#14141a] text-white',
  late: 'bg-[#52525b] text-white',
  half: 'bg-[#a1a1aa] text-white',
  absent: 'border border-black/20 text-fg',
  leave: 'bg-black/[0.06] text-fg-muted',
  off: 'bg-black/[0.03] text-fg-faint',
  holiday: 'bg-black/[0.03] text-fg-faint',
}

const WEEKDAYS = ['S', 'M', 'T', 'W', 'T', 'F', 'S']

/**
 * A single month's attendance as a 7-column calendar grid. Each day is toned by
 * its status; days can be clicked (e.g. to mark/override attendance). Days not
 * in `days` render as empty (no record). Reused by the user-detail and monthly
 * report pages.
 */
export function AttendanceCalendar({
  year,
  month,
  days,
  onDayClick,
}: {
  /** Full year, e.g. 2026. */
  year: number
  /** 1-12. */
  month: number
  days: CalendarDay[]
  onDayClick?: (day: CalendarDay | { date: string }) => void
}) {
  // Index days by their day-of-month for O(1) cell lookup.
  const byDay = useMemo(() => {
    const m = new Map<number, CalendarDay>()
    for (const d of days) {
      const day = Number(d.date.slice(8, 10))
      if (day) m.set(day, d)
    }
    return m
  }, [days])

  const firstWeekday = new Date(year, month - 1, 1).getDay() // 0=Sun
  const daysInMonth = new Date(year, month, 0).getDate()
  const cells: (number | null)[] = [
    ...Array<null>(firstWeekday).fill(null),
    ...Array.from({ length: daysInMonth }, (_, i) => i + 1),
  ]

  const pad2 = (n: number) => String(n).padStart(2, '0')

  return (
    <div>
      <div className="mb-2 grid grid-cols-7 gap-1.5 text-center text-[11px] font-semibold uppercase tracking-wide text-fg-faint">
        {WEEKDAYS.map((w, i) => (
          <span key={i}>{w}</span>
        ))}
      </div>
      <div className="grid grid-cols-7 gap-1.5">
        {cells.map((day, i) => {
          if (day == null) return <span key={`b${i}`} />
          const rec = byDay.get(day)
          const iso = `${year}-${pad2(month)}-${pad2(day)}`
          const style = rec ? STATUS_STYLE[rec.status] ?? 'bg-black/[0.04] text-fg-muted' : ''
          const clickable = !!onDayClick
          return (
            <button
              key={day}
              type="button"
              disabled={!clickable}
              onClick={() => onDayClick?.(rec ?? { date: iso })}
              title={rec?.title ?? iso}
              className={cn(
                'grid aspect-square place-items-center rounded-lg text-sm font-medium transition-colors',
                rec ? style : 'text-fg-faint hover:bg-black/[0.03]',
                clickable && 'cursor-pointer',
                !clickable && 'cursor-default',
              )}
            >
              {day}
            </button>
          )
        })}
      </div>
    </div>
  )
}

/** Legend describing the calendar's status tones. */
export function CalendarLegend({
  statuses = ['present', 'late', 'half', 'absent', 'leave'],
}: {
  statuses?: DayStatus[]
}) {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1.5 text-xs text-fg-muted">
      {statuses.map((s) => (
        <span key={s} className="flex items-center gap-1.5">
          <span
            className={cn(
              'grid h-4 w-4 place-items-center rounded',
              STATUS_STYLE[s] ?? 'bg-black/[0.04]',
            )}
          />
          <span className="capitalize">{s}</span>
        </span>
      ))}
    </div>
  )
}
