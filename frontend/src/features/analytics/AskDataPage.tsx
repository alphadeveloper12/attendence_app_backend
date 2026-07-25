import { useMemo, useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import type { ColumnDef } from '@tanstack/react-table'
import { useNavigate } from 'react-router-dom'
import { Search, Sparkles, CornerDownLeft } from 'lucide-react'
import { askData, type AskResult } from '../../lib/api/analytics'
import { PageHeader } from '../../components/ui/PageHeader'
import { DataTable } from '../../components/ui/DataTable'
import { GlassCard } from '../../components/ui/GlassCard'
import { Pill } from '../../components/ui/Pill'
import { cn } from '../../lib/utils'

/** Example prompts to seed the box — click to run one. */
const EXAMPLES = [
  'How many workers are on leave?',
  'Show me electricians at Site A',
  'Absent employees today',
  'List resigned staff',
]

/**
 * Ask the Data — a natural-language question box over the workforce. The server
 * parses it (LLM when enabled, else a keyword heuristic), echoes how it read the
 * question, and returns matching employees. Purely read-only.
 */
export default function AskDataPage() {
  const [q, setQ] = useState('')
  const navigate = useNavigate()

  const ask = useMutation({ mutationFn: (question: string) => askData(question) })
  const answer = ask.data

  function run(question: string) {
    const text = question.trim()
    if (!text) return
    setQ(text)
    ask.mutate(text)
  }

  const columns = useMemo<ColumnDef<AskResult, any>[]>(
    () => [
      {
        accessorKey: 'name',
        header: 'Employee',
        cell: (c) => (
          <div>
            <p className="font-semibold text-fg">{c.getValue<string>()}</p>
            <p className="text-xs text-fg-faint">#{c.row.original.badge}</p>
          </div>
        ),
      },
      { accessorKey: 'site', header: 'Site' },
      { accessorKey: 'department', header: 'Department' },
      { accessorKey: 'position', header: 'Position' },
      {
        accessorKey: 'status',
        header: 'Status',
        cell: (c) => <Pill tone="soft">{c.getValue<string>() || '—'}</Pill>,
      },
    ],
    [],
  )

  // Only the non-empty facets of how the question was understood.
  const understood = answer
    ? Object.entries(answer.as_understood).filter(([, v]) => v)
    : []

  return (
    <div>
      <PageHeader
        title="Ask the Data"
        subtitle="Ask a plain-English question about the workforce."
      />

      {/* Question box */}
      <GlassCard className="mb-4 p-2">
        <form
          onSubmit={(e) => {
            e.preventDefault()
            run(q)
          }}
          className="flex items-center gap-2"
        >
          <Search size={18} className="ml-2 shrink-0 text-fg-faint" />
          <input
            autoFocus
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="e.g. How many electricians are on leave at Site A?"
            className="h-11 flex-1 bg-transparent text-sm font-medium text-fg outline-none placeholder:font-normal placeholder:text-fg-faint"
          />
          <button
            type="submit"
            disabled={ask.isPending || !q.trim()}
            className="flex h-9 items-center gap-1.5 rounded-xl bg-[#14141a] px-4 text-sm font-semibold text-white transition-opacity hover:opacity-90 disabled:opacity-40"
          >
            {ask.isPending ? 'Asking…' : 'Ask'}
            <CornerDownLeft size={15} />
          </button>
        </form>
      </GlassCard>

      {/* Example chips (before first ask) */}
      {!answer && !ask.isPending && (
        <div className="mb-4 flex flex-wrap gap-2">
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              type="button"
              onClick={() => run(ex)}
              className="glass glass-hover rounded-full px-3.5 py-1.5 text-xs font-medium text-fg-muted"
            >
              {ex}
            </button>
          ))}
        </div>
      )}

      {ask.isError && (
        <GlassCard className="mb-4 p-4 text-sm text-fg">
          Couldn't answer that — {(ask.error as Error).message}
        </GlassCard>
      )}

      {answer && (
        <>
          {/* Answer summary */}
          <GlassCard className="mb-4 p-5">
            <div className="flex items-start gap-3">
              <span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-black/[0.05] text-fg">
                <Sparkles size={18} />
              </span>
              <div className="min-w-0">
                <p className="text-lg font-semibold text-fg">{answer.answer}</p>
                <div className="mt-2 flex flex-wrap items-center gap-1.5">
                  <span
                    className={cn(
                      'rounded-full px-2 py-0.5 text-[11px] font-semibold',
                      answer.engine === 'llm'
                        ? 'bg-[#14141a] text-white'
                        : 'bg-black/[0.05] text-fg-muted',
                    )}
                  >
                    {answer.engine === 'llm' ? 'AI parsed' : 'keyword match'}
                  </span>
                  {understood.map(([k, v]) => (
                    <Pill key={k} tone="soft">
                      {k}: {String(v)}
                    </Pill>
                  ))}
                </div>
              </div>
            </div>
          </GlassCard>

          <DataTable
            columns={columns}
            data={answer.results}
            pageSize={15}
            onRowClick={(r) => navigate(`/dashboard/user/${r.employee_id}-v2`)}
            emptyMessage="No employees matched your question."
          />
          {answer.count > answer.results.length && (
            <p className="mt-3 text-center text-xs text-fg-faint">
              Showing the first {answer.results.length} of {answer.count.toLocaleString()} matches.
            </p>
          )}
        </>
      )}
    </div>
  )
}
