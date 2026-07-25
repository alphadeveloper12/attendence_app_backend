import { motion, type Variant } from 'framer-motion'
import type { ReactNode } from 'react'

type Props = {
  children: ReactNode
  className?: string
  /** stagger delay in seconds */
  delay?: number
  /** travel distance in px */
  y?: number
  once?: boolean
}

const hidden: Variant = { opacity: 0, y: 24, filter: 'blur(6px)' }
const shown: Variant = { opacity: 1, y: 0, filter: 'blur(0px)' }

/** Scroll-into-view reveal with a soft blur/rise. Respects reduced-motion. */
export function Reveal({ children, className, delay = 0, y = 24, once = true }: Props) {
  return (
    <motion.div
      className={className}
      initial={hidden}
      whileInView={shown}
      viewport={{ once, margin: '-80px' }}
      transition={{ duration: 0.7, delay, ease: [0.22, 1, 0.36, 1], custom: y }}
    >
      {children}
    </motion.div>
  )
}

/** Container that staggers its <Reveal> children. */
export function RevealGroup({
  children,
  className,
  stagger = 0.08,
}: {
  children: ReactNode
  className?: string
  stagger?: number
}) {
  return (
    <motion.div
      className={className}
      initial="hidden"
      whileInView="shown"
      viewport={{ once: true, margin: '-80px' }}
      transition={{ staggerChildren: stagger }}
    >
      {children}
    </motion.div>
  )
}
