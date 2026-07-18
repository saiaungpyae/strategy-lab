import { useEffect, useRef } from 'react'

// Bridges the imperative canvas charts in lib/canvas.ts into React: clears the
// container and calls `draw` whenever `deps` change or the box is resized.
export default function CanvasBox({
  draw,
  deps,
  className,
}: {
  draw: (el: HTMLDivElement) => void
  deps: readonly unknown[]
  className?: string
}) {
  const ref = useRef<HTMLDivElement>(null)
  const drawRef = useRef(draw)
  drawRef.current = draw

  useEffect(() => {
    const el = ref.current
    if (!el) return
    let lastW = -1
    const render = () => {
      el.innerHTML = ''
      drawRef.current(el)
      lastW = el.clientWidth
    }
    render()
    const ro = new ResizeObserver(() => {
      if (el.clientWidth !== lastW) render()
    })
    ro.observe(el)
    return () => {
      ro.disconnect()
      el.innerHTML = ''
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  return <div ref={ref} className={className} />
}
