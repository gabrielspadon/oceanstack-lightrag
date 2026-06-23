import { lazy, Suspense } from 'react'

import GraphViewer from '@/features/GraphViewer'
import { useSettingsStore } from '@/stores/settings'

// Cosmos pulls in WebGL + DuckDB-wasm; lazy-load it so the default sigma path
// carries none of that weight and a load failure can never break the main bundle.
const GraphViewerCosmos = lazy(() => import('@/features/GraphViewerCosmos'))

/**
 * Chooses the graph renderer: sigma.js (default, code KG and small graphs) or
 * Cosmos.gl (GPU, large maritime graph). The toggle persists via the settings
 * store; sigma stays the fallback so the viewer is always usable.
 */
const GraphViewerSwitch = () => {
  const vizEngine = useSettingsStore.use.vizEngine()
  const setVizEngine = useSettingsStore.use.setVizEngine()

  return (
    <div className="relative h-full w-full">
      {vizEngine === 'cosmos' ? (
        <Suspense
          fallback={
            <div className="flex h-full w-full items-center justify-center text-muted-foreground">
              <p className="text-sm">Loading GPU graph…</p>
            </div>
          }
        >
          <GraphViewerCosmos />
        </Suspense>
      ) : (
        <GraphViewer />
      )}

      <button
        type="button"
        onClick={() => setVizEngine(vizEngine === 'cosmos' ? 'sigma' : 'cosmos')}
        className="bg-background/70 absolute right-2 bottom-2 z-20 rounded-md border px-2 py-1 text-xs font-medium backdrop-blur-lg"
        title="Switch graph renderer"
      >
        {vizEngine === 'cosmos' ? 'Sigma' : 'Cosmos (GPU)'}
      </button>
    </div>
  )
}

export default GraphViewerSwitch
