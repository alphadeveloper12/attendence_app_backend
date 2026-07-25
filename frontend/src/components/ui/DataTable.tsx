import { useState } from 'react'
import {
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  getPaginationRowModel,
  useReactTable,
  type ColumnDef,
  type SortingState,
} from '@tanstack/react-table'
import { ArrowUpDown, ArrowUp, ArrowDown, ChevronLeft, ChevronRight } from 'lucide-react'
import { GlassCard } from './GlassCard'
import { SkeletonRows } from './Skeleton'
import { EmptyState } from './EmptyState'
import { cn } from '../../lib/utils'

/**
 * Reusable data table built on TanStack Table. Client-side sorting + pagination
 * (fine for the analytics endpoints that return the full result set). Renders
 * its own loading (skeleton) and empty states inside a glass card.
 *
 * @typeParam T - row shape
 */
export function DataTable<T>({
  columns,
  data,
  loading = false,
  pageSize = 15,
  onRowClick,
  emptyMessage,
}: {
  columns: ColumnDef<T, any>[]
  data: T[]
  loading?: boolean
  pageSize?: number
  onRowClick?: (row: T) => void
  emptyMessage?: string
}) {
  const [sorting, setSorting] = useState<SortingState>([])

  const table = useReactTable({
    data,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    initialState: { pagination: { pageSize } },
  })

  if (loading) {
    return (
      <GlassCard className="p-4">
        <SkeletonRows rows={8} />
      </GlassCard>
    )
  }
  if (!data.length) {
    return (
      <GlassCard>
        <EmptyState message={emptyMessage} />
      </GlassCard>
    )
  }

  const { pageIndex, pageSize: ps } = table.getState().pagination
  const total = data.length
  const from = pageIndex * ps + 1
  const to = Math.min((pageIndex + 1) * ps, total)

  return (
    <GlassCard className="overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            {table.getHeaderGroups().map((hg) => (
              <tr key={hg.id} className="border-b border-black/10">
                {hg.headers.map((header) => {
                  const canSort = header.column.getCanSort()
                  const sorted = header.column.getIsSorted()
                  return (
                    <th
                      key={header.id}
                      className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-fg-faint"
                    >
                      {header.isPlaceholder ? null : (
                        <button
                          type="button"
                          disabled={!canSort}
                          onClick={header.column.getToggleSortingHandler()}
                          className={cn(
                            'inline-flex items-center gap-1.5',
                            canSort && 'cursor-pointer hover:text-fg',
                          )}
                        >
                          {flexRender(header.column.columnDef.header, header.getContext())}
                          {canSort &&
                            (sorted === 'asc' ? (
                              <ArrowUp size={13} />
                            ) : sorted === 'desc' ? (
                              <ArrowDown size={13} />
                            ) : (
                              <ArrowUpDown size={13} className="opacity-40" />
                            ))}
                        </button>
                      )}
                    </th>
                  )
                })}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.map((row) => (
              <tr
                key={row.id}
                onClick={onRowClick ? () => onRowClick(row.original) : undefined}
                className={cn(
                  'border-b border-black/[0.06] last:border-0 transition-colors',
                  onRowClick && 'cursor-pointer hover:bg-black/[0.03]',
                )}
              >
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id} className="px-4 py-3 text-fg">
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* pagination */}
      {table.getPageCount() > 1 && (
        <div className="flex items-center justify-between border-t border-black/10 px-4 py-3 text-sm text-fg-muted">
          <span>
            {from.toLocaleString()}–{to.toLocaleString()} of {total.toLocaleString()}
          </span>
          <div className="flex items-center gap-1">
            <PagerButton onClick={() => table.previousPage()} disabled={!table.getCanPreviousPage()}>
              <ChevronLeft size={16} />
            </PagerButton>
            <span className="px-2 text-xs">
              Page {pageIndex + 1} / {table.getPageCount()}
            </span>
            <PagerButton onClick={() => table.nextPage()} disabled={!table.getCanNextPage()}>
              <ChevronRight size={16} />
            </PagerButton>
          </div>
        </div>
      )}
    </GlassCard>
  )
}

function PagerButton({
  children,
  onClick,
  disabled,
}: {
  children: React.ReactNode
  onClick: () => void
  disabled?: boolean
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className="grid h-8 w-8 place-items-center rounded-lg text-fg-muted transition-colors hover:bg-black/[0.05] hover:text-fg disabled:cursor-not-allowed disabled:opacity-40"
    >
      {children}
    </button>
  )
}
