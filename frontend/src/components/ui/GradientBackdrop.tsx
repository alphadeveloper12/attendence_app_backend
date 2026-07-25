/**
 * Fixed page backdrop: bright, clean white canvas with only the faintest neutral
 * soft shapes so the frosted glass has a little something to refract. Minimalist.
 */
export function GradientBackdrop() {
  return (
    <div aria-hidden className="fixed inset-0 -z-10 overflow-hidden bg-[#f4f5f7]">
      {/* soft white sheen top */}
      <div className="absolute inset-x-0 -top-40 h-[560px] bg-[radial-gradient(60%_100%_at_50%_0%,#ffffff,transparent_70%)]" />
      {/* barely-there neutral depth blobs */}
      <div className="absolute -top-40 -left-32 h-[520px] w-[620px] rounded-full bg-[#e4e7ec]/60 blur-[150px]" />
      <div className="absolute bottom-[-10%] right-[-8%] h-[560px] w-[640px] rounded-full bg-[#e7e4ea]/55 blur-[160px]" />
      <div className="absolute top-1/2 left-1/4 h-[420px] w-[520px] rounded-full bg-[#eef0f3]/60 blur-[150px]" />
    </div>
  )
}
