import { lazy, Suspense } from 'react'

import GraphViewer from '@/features/GraphViewer'
import { useSettingsStore } from '@/stores/settings'
import { useBackendState } from '@/stores/state'

// Cosmos (WebGL/DuckDB-free) and the deck.gl map are lazy-loaded so the default
// sigma path carries none of their weight and a load failure can't break it.
const GraphViewerCosmos = lazy(() => import('@/features/GraphViewerCosmos'))
const MapViewer = lazy(() => import('@/features/MapViewer'))

const LABELS = {
  sigma: 'Sigma',
  cosmos: 'Cosmos (GPU)',
  map: 'Map'
} as const
type Engine = keyof typeof LABELS

const Loading = () => (
  <div className="text-muted-foreground flex h-full w-full items-center justify-center">
    <p className="text-sm">Loading…</p>
  </div>
)

/**
 * Chooses the view: sigma.js (default), Cosmos.gl (GPU, large graphs), or the
 * deck.gl geographic map. Sigma and Cosmos are both graph viewers and apply to
 * every workspace. The map is geographic, so it is offered only for the maritime
 * knowledge graph (whose entities carry coordinates); the code KG has no
 * geography and never shows it. The engine setting persists in localStorage
 * shared by both servers' webui, so the stored value is clamped here rather than
 * trusted — a map selection made on maritime cannot strand the code KG.
 */
const GraphViewerSwitch = () => {
  const vizEngine = useSettingsStore.use.vizEngine()
  const setVizEngine = useSettingsStore.use.setVizEngine()
  const status = useBackendState.use.status()

  const isMaritime = (status?.configuration?.workspace ?? '').includes('maritime')
  const engines: Engine[] = isMaritime ? ['sigma', 'cosmos', 'map'] : ['sigma', 'cosmos']
  const engine: Engine = engines.includes(vizEngine) ? vizEngine : 'sigma'
  const next = engines[(engines.indexOf(engine) + 1) % engines.length]

  return (
    <div className="relative h-full w-full">
      {engine === 'cosmos' ? (
        <Suspense fallback={<Loading />}>
          <GraphViewerCosmos />
        </Suspense>
      ) : engine === 'map' ? (
        <Suspense fallback={<Loading />}>
          <MapViewer />
        </Suspense>
      ) : (
        <GraphViewer />
      )}

      {engines.length > 1 && (
        <button
          type="button"
          onClick={() => setVizEngine(next)}
          className="bg-background/70 absolute top-2 right-2 z-30 rounded-md border px-2 py-1 text-xs font-medium backdrop-blur-lg"
          title="Switch view"
        >
          View: {LABELS[engine]} → {LABELS[next]}
        </button>
      )}
    </div>
  )
}

export default GraphViewerSwitch
