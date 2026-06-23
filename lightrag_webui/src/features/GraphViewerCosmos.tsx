import { useEffect, useMemo, useRef } from 'react'
import { Graph } from '@cosmos.gl/graph'

import useLightrangeGraph from '@/hooks/useLightragGraph'
import { useGraphStore } from '@/stores/graph'
import { useSettingsStore } from '@/stores/settings'

/**
 * GPU (WebGL) graph viewer for the large maritime knowledge graph.
 *
 * Uses cosmos.gl's low-level Graph (flat Float32Array buffers, no DuckDB-wasm),
 * so it renders far past sigma.js's interactive ceiling and starts reliably. It
 * reuses the shared fetch hook — the same rawGraph the sigma viewer consumes —
 * and maps it to id->index point/link buffers. Sigma stays the default and the
 * code-KG viewer.
 */

const ENCOUNTER_PATTERN = /encounter|co-location|rendezvous/i

const isEncounterEdge = (edgeType: string | undefined, keywords: unknown): boolean => {
  if (edgeType && ENCOUNTER_PATTERN.test(edgeType)) return true
  return typeof keywords === 'string' && ENCOUNTER_PATTERN.test(keywords)
}

// Parse a #rrggbb / #rgb string into normalised WebGL RGBA (0-1). Falls back to a
// neutral blue for names or unparseable values.
const toRgba = (color: string | undefined): [number, number, number, number] => {
  if (color && /^#([0-9a-f]{6})$/i.test(color)) {
    const n = parseInt(color.slice(1), 16)
    return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255, 1]
  }
  if (color && /^#([0-9a-f]{3})$/i.test(color)) {
    const r = parseInt(color[1], 16) / 15
    const g = parseInt(color[2], 16) / 15
    const b = parseInt(color[3], 16) / 15
    return [r, g, b, 1]
  }
  return [0.53, 0.67, 0.8, 1]
}

const GraphViewerCosmos = () => {
  // Drive the shared data fetch (queryLabel -> rawGraph). No sigma context needed;
  // the hook populates the store via effects.
  useLightrangeGraph()
  const rawGraph = useGraphStore.use.rawGraph()
  const hideEncounterEdges = useSettingsStore.use.hideEncounterEdges()

  const containerRef = useRef<HTMLDivElement | null>(null)
  const graphRef = useRef<Graph | null>(null)

  const buffers = useMemo(() => {
    if (!rawGraph || rawGraph.nodes.length === 0) return null
    const n = rawGraph.nodes.length
    const positions = new Float32Array(n * 2)
    const colors = new Float32Array(n * 4)
    const sizes = new Float32Array(n)
    const idToIndex = new Map<string, number>()

    rawGraph.nodes.forEach((node, i) => {
      idToIndex.set(node.id, i)
      // Seed positions (the fetch hook assigns them); the force simulation
      // settles the layout from here.
      positions[i * 2] = node.x
      positions[i * 2 + 1] = node.y
      const [r, g, b, a] = toRgba(node.color)
      colors[i * 4] = r
      colors[i * 4 + 1] = g
      colors[i * 4 + 2] = b
      colors[i * 4 + 3] = a
      sizes[i] = Math.max(2, node.size ?? 4)
    })

    const linkPairs: number[] = []
    for (const e of rawGraph.edges) {
      if (hideEncounterEdges && isEncounterEdge(e.type, e.properties?.keywords)) continue
      const s = idToIndex.get(e.source)
      const t = idToIndex.get(e.target)
      if (s !== undefined && t !== undefined) linkPairs.push(s, t)
    }

    return { positions, colors, sizes, links: new Float32Array(linkPairs) }
  }, [rawGraph, hideEncounterEdges])

  // Create the Graph once the container exists; tear it down on unmount.
  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    const graph = new Graph(container, {
      backgroundColor: [0, 0, 0, 0],
      pointSizeScale: 1,
      linkWidthScale: 0.5,
      simulationGravity: 0.1,
      simulationRepulsion: 0.4,
      simulationLinkDistance: 8,
      fitViewOnInit: true,
      enableDrag: true,
      scalePointsOnZoom: true
    })
    graphRef.current = graph
    return () => {
      graph.destroy()
      graphRef.current = null
    }
  }, [])

  // Push buffer updates whenever the data changes.
  useEffect(() => {
    const graph = graphRef.current
    if (!graph || !buffers) return
    graph.setPointPositions(buffers.positions)
    graph.setPointColors(buffers.colors)
    graph.setPointSizes(buffers.sizes)
    graph.setLinks(buffers.links)
    graph.render()
  }, [buffers])

  return (
    <div className="relative h-full w-full">
      <div ref={containerRef} className="h-full w-full" />
      {(!rawGraph || rawGraph.nodes.length === 0) && (
        <div className="text-muted-foreground absolute inset-0 flex items-center justify-center">
          <p className="text-sm">Select a label to load the graph.</p>
        </div>
      )}
    </div>
  )
}

export default GraphViewerCosmos
