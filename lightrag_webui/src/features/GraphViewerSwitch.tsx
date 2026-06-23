import { lazy, Suspense } from 'react'

import GraphViewer from '@/features/GraphViewer'
import { useSettingsStore } from '@/stores/settings'

// Cosmos (WebGL/DuckDB-free) and the deck.gl map are lazy-loaded so the default
// sigma path carries none of their weight and a load failure can't break it.
const GraphViewerCosmos = lazy(() => import('@/features/GraphViewerCosmos'))
const MapViewer = lazy(() => import('@/features/MapViewer'))

const ENGINES = ['sigma', 'cosmos', 'map'] as const
const LABELS: Record<(typeof ENGINES)[number], string> = {
  sigma: 'Sigma',
  cosmos: 'Cosmos (GPU)',
  map: 'Map'
}

const Loading = () => (
  <div className="text-muted-foreground flex h-full w-full items-center justify-center">
    <p className="text-sm">Loading…</p>
  </div>
)

/**
 * Chooses the view: sigma.js (default, code KG and small graphs), Cosmos.gl (GPU,
 * large maritime graph), or the deck.gl geographic map. The choice persists via
 * the settings store; sigma stays the fallback so the viewer is always usable.
 */
const GraphViewerSwitch = () => {
  const vizEngine = useSettingsStore.use.vizEngine()
  const setVizEngine = useSettingsStore.use.setVizEngine()
  const next = ENGINES[(ENGINES.indexOf(vizEngine) + 1) % ENGINES.length]

  return (
    <div className="relative h-full w-full">
      {vizEngine === 'cosmos' ? (
        <Suspense fallback={<Loading />}>
          <GraphViewerCosmos />
        </Suspense>
      ) : vizEngine === 'map' ? (
        <Suspense fallback={<Loading />}>
          <MapViewer />
        </Suspense>
      ) : (
        <GraphViewer />
      )}

      <button
        type="button"
        onClick={() => setVizEngine(next)}
        className="bg-background/70 absolute top-2 right-2 z-30 rounded-md border px-2 py-1 text-xs font-medium backdrop-blur-lg"
        title="Switch view"
      >
        View: {LABELS[vizEngine]} → {LABELS[next]}
      </button>
    </div>
  )
}

export default GraphViewerSwitch
