import { useMemo } from 'react'
import { Cosmograph, CosmographProvider, CosmographSearch } from '@cosmograph/react'

import useLightrangeGraph from '@/hooks/useLightragGraph'
import { useGraphStore } from '@/stores/graph'
import { useSettingsStore } from '@/stores/settings'

/**
 * GPU (WebGL) graph viewer for the large maritime knowledge graph.
 *
 * Cosmos.gl renders far past sigma.js's interactive ceiling, so the maritime
 * workspace (hundreds of thousands of vessel/port nodes) stays navigable when
 * the server node cap is raised. It reuses the shared fetch hook — the same
 * rawGraph the sigma viewer consumes — and transforms it into Cosmograph's
 * tabular point/link records. Sigma remains the default and the code-KG viewer.
 */

type CosmosPoint = {
  id: string
  label: string
  type: string
  degree: number
}

type CosmosLink = {
  source: string
  target: string
}

// Encounter (vessel-vessel) edges are the bulk of the maritime graph and dominate
// the view; this matches the keywords the rebuild assigns to that layer so they
// can be hidden, leaving the structural frequents (vessel-port) and corridor
// (port-port) edges.
const ENCOUNTER_PATTERN = /encounter|co-location|rendezvous/i

const isEncounterEdge = (edgeType: string | undefined, keywords: unknown): boolean => {
  if (edgeType && ENCOUNTER_PATTERN.test(edgeType)) return true
  return typeof keywords === 'string' && ENCOUNTER_PATTERN.test(keywords)
}

const GraphViewerCosmos = () => {
  // Drive the shared data fetch (queryLabel -> rawGraph). No sigma context needed;
  // the hook populates the store via effects.
  useLightrangeGraph()
  const rawGraph = useGraphStore.use.rawGraph()
  const hideEncounterEdges = useSettingsStore.use.hideEncounterEdges()

  const { points, links } = useMemo(() => {
    if (!rawGraph) return { points: [] as CosmosPoint[], links: [] as CosmosLink[] }

    const points: CosmosPoint[] = rawGraph.nodes.map((n) => ({
      id: n.id,
      label: n.labels?.[0] ?? n.id,
      type: (n.properties?.entity_type as string | undefined) ?? 'unknown',
      degree: n.degree ?? 0
    }))

    const ids = new Set(points.map((p) => p.id))
    const links: CosmosLink[] = rawGraph.edges
      .filter((e) => ids.has(e.source) && ids.has(e.target))
      .filter((e) => !(hideEncounterEdges && isEncounterEdge(e.type, e.properties?.keywords)))
      .map((e) => ({ source: e.source, target: e.target }))

    return { points, links }
  }, [rawGraph, hideEncounterEdges])

  if (!rawGraph || points.length === 0) {
    return (
      <div className="flex h-full w-full items-center justify-center text-muted-foreground">
        <p className="text-sm">Select a label to load the graph.</p>
      </div>
    )
  }

  return (
    <div className="relative h-full w-full">
      <CosmographProvider>
        <Cosmograph
          points={points}
          links={links}
          pointIdBy="id"
          pointLabelBy="label"
          pointColorBy="type"
          pointSizeBy="degree"
          linkSourceBy="source"
          linkTargetBy="target"
          backgroundColor="#00000000"
          style={{ width: '100%', height: '100%' }}
        />
        <div className="absolute top-2 left-2 z-10">
          <CosmographSearch />
        </div>
      </CosmographProvider>
    </div>
  )
}

export default GraphViewerCosmos
