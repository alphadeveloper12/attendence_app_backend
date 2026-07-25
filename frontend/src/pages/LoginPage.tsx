import { useRef, useState, type FormEvent } from 'react'
import { motion } from 'framer-motion'
import {
  Rocket, Eye, EyeOff, AlertCircle, Loader2, Smartphone, ArrowLeft, Lock, User,
} from 'lucide-react'
import { GradientBackdrop } from '../components/ui/GradientBackdrop'
import { GlassCard } from '../components/ui/GlassCard'
import { sessionLogin } from '../lib/auth'
import { cn } from '../lib/utils'

export default function LoginPage() {
  const [showPw, setShowPw] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const userRef = useRef<HTMLInputElement>(null)

  async function onSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault()
    if (loading) return
    setError('')
    setLoading(true)
    const form = new FormData(e.currentTarget)
    const res = await sessionLogin(
      String(form.get('username') || ''),
      String(form.get('password') || ''),
    )
    if (res.ok) {
      // Land in the new React dashboard (not the legacy /dashboard/ the server
      // returns). Full navigation so the SPA remounts with a fresh session.
      window.location.href = '/dashboard-v2/'
    } else {
      setError(res.error)
      setLoading(false)
    }
  }

  return (
    <div className="relative flex min-h-screen items-center justify-center px-5 py-12">
      <GradientBackdrop />

      {/* back to home */}
      <a
        href="/home-v2/"
        className="absolute left-5 top-5 flex items-center gap-1.5 rounded-full px-3 py-2 text-sm font-medium text-fg-muted transition-colors hover:bg-black/[0.04] hover:text-fg sm:left-8 sm:top-8"
      >
        <ArrowLeft size={16} /> Back to home
      </a>

      <motion.div
        initial={{ opacity: 0, y: 20, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
        className="w-full max-w-[430px]"
      >
        <GlassCard strong className="p-8 sm:p-10">
          {/* brand */}
          <div className="flex flex-col items-center text-center">
            <span className="grid h-14 w-14 place-items-center rounded-2xl bg-[#14141a] shadow-[0_12px_30px_-10px_rgba(17,17,26,0.5)]">
              <Rocket size={26} className="text-white" />
            </span>
            <h1 className="mt-5 text-2xl font-extrabold text-gradient">Welcome back</h1>
            <p className="mt-1.5 text-sm text-fg-muted">
              Sign in to your RocketAttendance console
            </p>
          </div>

          {/* error */}
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -6 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-6 flex items-center gap-2.5 rounded-xl border border-black/10 bg-black/[0.03] px-4 py-3 text-sm font-medium text-fg"
            >
              <AlertCircle size={17} className="shrink-0" />
              {error}
            </motion.div>
          )}

          {/* form */}
          <form onSubmit={onSubmit} className="mt-7 flex flex-col gap-5">
            <Field label="Username" htmlFor="username" icon={<User size={16} />}>
              <input
                ref={userRef}
                id="username"
                name="username"
                type="text"
                required
                autoFocus
                autoComplete="username"
                placeholder="Enter your username"
                className={inputClass}
              />
            </Field>

            <Field label="Password" htmlFor="password" icon={<Lock size={16} />}>
              <div className="relative">
                <input
                  id="password"
                  name="password"
                  type={showPw ? 'text' : 'password'}
                  required
                  autoComplete="current-password"
                  placeholder="Enter your password"
                  className={cn(inputClass, 'pr-11')}
                />
                <button
                  type="button"
                  onClick={() => setShowPw((v) => !v)}
                  aria-label={showPw ? 'Hide password' : 'Show password'}
                  className="absolute right-2 top-1/2 grid h-8 w-8 -translate-y-1/2 place-items-center rounded-lg text-fg-faint transition-colors hover:bg-black/[0.05] hover:text-fg"
                >
                  {showPw ? <EyeOff size={17} /> : <Eye size={17} />}
                </button>
              </div>
            </Field>

            <button
              type="submit"
              disabled={loading}
              className="mt-1 flex h-12 w-full items-center justify-center gap-2 rounded-full bg-[#14141a] text-[15px] font-semibold text-white shadow-[0_12px_34px_-12px_rgba(17,17,26,0.5)] transition-all hover:bg-black hover:-translate-y-0.5 disabled:cursor-not-allowed disabled:opacity-70 disabled:hover:translate-y-0"
            >
              {loading ? (
                <>
                  <Loader2 size={18} className="animate-spin" /> Signing in…
                </>
              ) : (
                'Sign in'
              )}
            </button>
          </form>

          {/* divider */}
          <div className="my-7 flex items-center gap-3 text-xs text-fg-faint">
            <span className="h-px flex-1 bg-black/10" />
            OR
            <span className="h-px flex-1 bg-black/10" />
          </div>

          {/* download app */}
          <a
            href="/dashboard/downloads/"
            className="flex h-12 w-full items-center justify-center gap-2 rounded-full glass text-[15px] font-semibold text-fg transition-all hover:-translate-y-0.5"
          >
            <Smartphone size={18} /> Download mobile app
          </a>
        </GlassCard>

        <p className="mt-6 text-center text-xs text-fg-faint">
          Protected access · UAE-region hosting · encrypted at rest
        </p>
      </motion.div>
    </div>
  )
}

const inputClass =
  'w-full rounded-xl border border-black/10 bg-white/60 px-4 py-3 text-[15px] font-medium text-fg ' +
  'placeholder:font-normal placeholder:text-fg-faint outline-none transition-all ' +
  'focus:border-black/25 focus:bg-white focus:ring-2 focus:ring-black/10'

function Field({
  label,
  htmlFor,
  icon,
  children,
}: {
  label: string
  htmlFor: string
  icon: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <div>
      <label
        htmlFor={htmlFor}
        className="mb-2 flex items-center gap-1.5 text-xs font-bold uppercase tracking-wide text-fg-muted"
      >
        {icon} {label}
      </label>
      {children}
    </div>
  )
}
