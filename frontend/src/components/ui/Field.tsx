import { Search } from 'lucide-react'
import { Select as AntSelect, DatePicker as AntDatePicker, TimePicker as AntTimePicker } from 'antd'
import dayjs from 'dayjs'
import customParseFormat from 'dayjs/plugin/customParseFormat'
import { cn } from '../../lib/utils'

// Needed so dayjs can parse "HH:mm" time strings back into dayjs objects.
dayjs.extend(customParseFormat)

/**
 * Shared form controls. Dropdowns and date/time pickers are Ant Design
 * components themed (via ConfigProvider in main.tsx) to match our monochrome
 * light-glass system — so they look consistent and premium everywhere.
 *
 * All popups render into the trigger's parent (not document.body) so they stay
 * clickable inside Radix modals, which set `pointer-events: none` on the body.
 */

/** Route antd popups to the trigger's parent (see note above). */
const popupToParent = (trigger: HTMLElement) => (trigger.parentElement as HTMLElement) ?? document.body

export interface Option {
  value: string
  label: string
}

/** Themed single-select dropdown. Empty value shows the placeholder. */
export function Select({
  value,
  onChange,
  options,
  placeholder,
  className,
  'aria-label': ariaLabel,
}: {
  value: string
  onChange: (value: string) => void
  options: Option[]
  placeholder?: string
  className?: string
  'aria-label'?: string
}) {
  return (
    <div className={className}>
      <AntSelect
        aria-label={ariaLabel}
        className="w-full"
        value={value === '' ? undefined : value}
        onChange={(v) => onChange((v as string) ?? '')}
        options={options}
        placeholder={placeholder}
        // Placeholder-style selects (no explicit "all" option) get a clear button
        // so the user can return to the all/placeholder state.
        allowClear={!!placeholder}
        getPopupContainer={popupToParent}
      />
    </div>
  )
}

/** Themed date picker. Value + onChange use ISO `YYYY-MM-DD` strings. */
export function DatePicker({
  value,
  onChange,
  className,
  placeholder = 'Select date',
  maxToday = false,
  'aria-label': ariaLabel,
}: {
  value: string
  onChange: (value: string) => void
  className?: string
  placeholder?: string
  /** Disable future dates (e.g. an "as of" historical picker). */
  maxToday?: boolean
  'aria-label'?: string
}) {
  return (
    <div className={className}>
      <AntDatePicker
        aria-label={ariaLabel}
        className="w-full"
        value={value ? dayjs(value) : null}
        onChange={(d) => onChange(d ? d.format('YYYY-MM-DD') : '')}
        format="DD MMM YYYY"
        placeholder={placeholder}
        disabledDate={maxToday ? (d) => d.isAfter(dayjs(), 'day') : undefined}
        getPopupContainer={popupToParent}
      />
    </div>
  )
}

/** Themed time picker. Value + onChange use `HH:mm` strings. */
export function TimePicker({
  value,
  onChange,
  className,
  placeholder = 'Select time',
  'aria-label': ariaLabel,
}: {
  value: string
  onChange: (value: string) => void
  className?: string
  placeholder?: string
  'aria-label'?: string
}) {
  return (
    <div className={className}>
      <AntTimePicker
        aria-label={ariaLabel}
        className="w-full"
        value={value ? dayjs(value, 'HH:mm') : null}
        onChange={(d) => onChange(d ? d.format('HH:mm') : '')}
        format="HH:mm"
        minuteStep={5}
        needConfirm={false}
        placeholder={placeholder}
        getPopupContainer={popupToParent}
      />
    </div>
  )
}

/** Search input with a leading icon (native — not a picker). */
export function SearchInput({
  value,
  onChange,
  placeholder = 'Search…',
  className,
}: {
  value: string
  onChange: (value: string) => void
  placeholder?: string
  className?: string
}) {
  const base =
    'h-10 rounded-xl border border-black/10 bg-white/60 text-sm font-medium text-fg ' +
    'outline-none transition-all focus:border-black/25 focus:bg-white focus:ring-2 focus:ring-black/10'
  return (
    <div className={cn('relative', className)}>
      <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-fg-faint" />
      <input
        type="search"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className={cn(base, 'w-full pl-9 pr-3 placeholder:font-normal placeholder:text-fg-faint')}
      />
    </div>
  )
}
