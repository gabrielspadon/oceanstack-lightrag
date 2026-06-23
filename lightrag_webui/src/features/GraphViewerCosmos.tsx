import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Graph } from '@cosmos.gl/graph'
import Graphology from 'graphology'
import louvain from 'graphology-communities-louvain'
import { MaximizeIcon } from 'lucide-react'

import useLightrangeGraph from '@/hooks/useLightragGraph'
import GraphLabels from '@/components/graph/GraphLabels'
import Legend from '@/components/graph/Legend'
import LegendButton from '@/components/graph/LegendButton'
import PropertiesView from '@/components/graph/PropertiesView'
import Button from '@/components/ui/Button'
import { controlButtonVariant } from '@/lib/constants'
import { useGraphStore, type RawGraph } from '@/stores/graph'
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

// WebGL RGBA (0-1) palette cycled across Louvain communities.
const COMMUNITY_PALETTE: [number, number, number, number][] = [
  [0.9, 0.3, 0.24, 1],
  [0.18, 0.55, 0.86, 1],
  [0.18, 0.74, 0.41, 1],
  [0.95, 0.61, 0.07, 1],
  [0.61, 0.35, 0.71, 1],
  [0.1, 0.74, 0.7, 1],
  [0.95, 0.77, 0.06, 1],
  [0.91, 0.3, 0.59, 1],
  [0.4, 0.5, 0.55, 1],
  [0.55, 0.76, 0.29, 1],
  [0.55, 0.45, 0.36, 1],
  [0.3, 0.65, 0.9, 1]
]

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
  const setSelectedNode = useGraphStore.use.setSelectedNode()
  const setFocusedNode = useGraphStore.use.setFocusedNode()
  const hideEncounterEdges = useSettingsStore.use.hideEncounterEdges()
  const showLegend = useSettingsStore.use.showLegend()
  const showPropertyPanel = useSettingsStore.use.showPropertyPanel()
  const colorByCommunity = useSettingsStore.use.colorByCommunity()
  const setColorByCommunity = useSettingsStore.use.setColorByCommunity()

  const containerRef = useRef<HTMLDivElement | null>(null)
  const graphRef = useRef<Graph | null>(null)
  const [search, setSearch] = useState('')

  // Latest graph for the cosmos click/hover callbacks, which are bound once at
  // Graph creation; the ref keeps them reading current data after a refetch.
  const rawGraphRef = useRef<RawGraph | null>(rawGraph)
  useEffect(() => {
    rawGraphRef.current = rawGraph
  }, [rawGraph])

  const handleFit = useCallback(() => graphRef.current?.fitView(), [])

  // label -> point index, for the search box: typing a node label zooms to it.
  const labelToIndex = useMemo(() => {
    const m = new Map<string, number>()
    rawGraph?.nodes.forEach((node, i) => m.set(node.labels?.[0] ?? node.id, i))
    return m
  }, [rawGraph])

  const handleSearch = useCallback(
    (label: string) => {
      const idx = labelToIndex.get(label)
      if (idx === undefined) return
      graphRef.current?.zoomToPointByIndex(idx)
      const node = rawGraph?.nodes[idx]
      if (node) setSelectedNode(node.id)
    },
    [labelToIndex, setSelectedNode, rawGraph]
  )

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
      sizes[i] = Math.max(2, node.size ?? 4)
    })

    // Build the encounter-filtered edge set plus a graphology graph for Louvain
    // community detection — the densest clusters share a colour so the structure
    // of the maritime graph reads at a glance.
    const linkPairs: number[] = []
    const g = new Graphology({ type: 'undirected' })
    rawGraph.nodes.forEach((node) => g.addNode(node.id))
    for (const e of rawGraph.edges) {
      if (hideEncounterEdges && isEncounterEdge(e.type, e.properties?.keywords)) continue
      const s = idToIndex.get(e.source)
      const t = idToIndex.get(e.target)
      if (s === undefined || t === undefined) continue
      linkPairs.push(s, t)
      if (e.source !== e.target && !g.hasEdge(e.source, e.target)) g.addEdge(e.source, e.target)
    }

    // Default to the same entity-type colours as the sigma viewer (so the Legend
    // matches); the community palette is opt-in via the colorByCommunity setting.
    const communities: Record<string, number> =
      colorByCommunity && g.size > 0 ? louvain(g) : {}
    rawGraph.nodes.forEach((node, i) => {
      const community = communities[node.id]
      const [r, gg, b, a] =
        community === undefined
          ? toRgba(node.color)
          : COMMUNITY_PALETTE[community % COMMUNITY_PALETTE.length]
      colors[i * 4] = r
      colors[i * 4 + 1] = gg
      colors[i * 4 + 2] = b
      colors[i * 4 + 3] = a
    })

    return { positions, colors, sizes, links: new Float32Array(linkPairs) }
  }, [rawGraph, hideEncounterEdges, colorByCommunity])

  // Create the Graph lazily and push buffers once its WebGL device is ready.
  // The device init is async, so on a re-mount (data already cached) we must wait
  // for `ready` before setData — otherwise the graph comes back blank after a
  // viewer toggle.
  useEffect(() => {
    const container = containerRef.current
    if (!container || !buffers) return
    let cancelled = false
    let graph = graphRef.current
    if (!graph) {
      graph = new Graph(container, {
        backgroundColor: [0, 0, 0, 0],
        pointSizeScale: 1,
        linkWidthScale: 0.5,
        simulationGravity: 0.1,
        simulationRepulsion: 0.4,
        simulationLinkDistance: 8,
        fitViewOnInit: true,
        enableDrag: true,
        scalePointsOnZoom: true,
        // Match the sigma viewer's interactions: click opens the PropertiesView
        // panel (the real per-entity data); hover focuses the node and rings it.
        onClick: (index) => {
          const id =
            index !== undefined ? (rawGraphRef.current?.nodes[index]?.id ?? null) : null
          setSelectedNode(id)
        },
        onPointMouseOver: (index) => {
          setFocusedNode(rawGraphRef.current?.nodes[index]?.id ?? null)
          graphRef.current?.setConfigPartial({ outlinedPointIndices: [index] })
        },
        onPointMouseOut: () => {
          setFocusedNode(null)
          graphRef.current?.setConfigPartial({ outlinedPointIndices: undefined })
        }
      })
      graphRef.current = graph
    }
    const g = graph
    g.ready.then(() => {
      if (cancelled) return
      g.setPointPositions(buffers.positions)
      g.setPointColors(buffers.colors)
      g.setPointSizes(buffers.sizes)
      g.setLinks(buffers.links)
      g.render()
    })
    return () => {
      cancelled = true
    }
  }, [buffers, setSelectedNode, setFocusedNode])

  // Release the WebGL context only on unmount.
  useEffect(
    () => () => {
      graphRef.current?.destroy()
      graphRef.current = null
    },
    []
  )

  return (
    <div className="relative h-full w-full">
      <div ref={containerRef} className="h-full w-full" />

      {showPropertyPanel && (
        <div className="absolute top-2 right-2 z-10">
          <PropertiesView />
        </div>
      )}

      <div className="absolute top-2 left-2 flex items-start gap-2">
        <GraphLabels />
        <input
          list="cosmos-node-labels"
          value={search}
          onChange={(e) => {
            setSearch(e.target.value)
            handleSearch(e.target.value)
          }}
          placeholder="Search node…"
          className="bg-background/70 h-9 w-44 rounded-md border px-2 text-xs backdrop-blur-lg"
        />
        <datalist id="cosmos-node-labels">
          {Array.from(labelToIndex.keys())
            .slice(0, 2000)
            .map((label) => (
              <option key={label} value={label} />
            ))}
        </datalist>
      </div>

      <div className="bg-background/60 absolute bottom-2 left-2 flex flex-col rounded-xl border-2 backdrop-blur-lg">
        <Button
          size="icon"
          variant={controlButtonVariant}
          onClick={handleFit}
          tooltip="Fit view"
        >
          <MaximizeIcon />
        </Button>
        <LegendButton />
      </div>

      <label className="bg-background/60 absolute bottom-2 left-14 z-10 flex items-center gap-1 rounded-md px-2 py-1 text-xs backdrop-blur-lg">
        <input
          type="checkbox"
          checked={colorByCommunity}
          onChange={(e) => setColorByCommunity(e.target.checked)}
        />
        Color by community
      </label>

      {showLegend && (
        <div className="absolute right-2 bottom-10 z-0">
          <Legend className="bg-background/60 backdrop-blur-lg" />
        </div>
      )}

      {(!rawGraph || rawGraph.nodes.length === 0) && (
        <div className="text-muted-foreground absolute inset-0 flex items-center justify-center">
          <p className="text-sm">Select a label to load the graph.</p>
        </div>
      )}
    </div>
  )
}

export default GraphViewerCosmos
