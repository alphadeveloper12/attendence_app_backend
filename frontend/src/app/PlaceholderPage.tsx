import { GlassCard } from '../components/ui/GlassCard'

/**
 * Temporary stand-in for dashboard pages not yet migrated. Replaced page by
 * page across the migration phases.
 */
export function PlaceholderPage({ title }: { title: string }) {
  return (
    <GlassCard className="grid place-items-center p-16 text-center">
      <div>
        <h2 className="text-xl font-bold text-fg">{title}</h2>
        <p className="mt-2 text-sm text-fg-muted">This page is being migrated to React.</p>
      </div>
    </GlassCard>
  )
}
