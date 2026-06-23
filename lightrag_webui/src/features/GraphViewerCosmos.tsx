import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Graph } from '@cosmos.gl/graph'
import Graphology from 'graphology'
import louvain from 'graphology-communities-louvain'
import { circular, circlepack, random } from 'graphology-layout'
import forceLayout from 'graphology-layout-force'
import forceAtlas2 from 'graphology-layout-forceatlas2'
import noverlap from 'graphology-layout-noverlap'
import {
  MaximizeIcon,
  MinimizeIcon,
  ZoomInIcon,
  ZoomOutIcon,
  FullscreenIcon,
  GripIcon,
  PlayIcon,
  PauseIcon,
  RotateCcwIcon,
  RotateCwIcon
} from 'lucide-react'
import type { GraphSearchOption, OptionItem } from '@react-sigma/graph-search'

import useLightrangeGraph from '@/hooks/useLightragGraph'
import GraphLabels from '@/components/graph/GraphLabels'
import GraphSearch from '@/components/graph/GraphSearch'
import Legend from '@/components/graph/Legend'
import LegendButton from '@/components/graph/LegendButton'
import PropertiesView from '@/components/graph/PropertiesView'
import Settings from '@/components/graph/Settings'
import SettingsDisplay from '@/components/graph/SettingsDisplay'
import Button from '@/components/ui/Button'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/Popover'
import { Command, CommandGroup, CommandItem, CommandList } from '@/components/ui/Command'
import { controlButtonVariant } from '@/lib/constants'
import { useGraphStore, type RawGraph } from '@/stores/graph'
import { useSettingsStore } from '@/stores/settings'

import '@react-sigma/graph-search/lib/style.css'

// The same layout algorithms the sigma viewer offers, computed on the graphology
// graph and applied to the cosmos position buffer. Force layouts iterate from the
// current positions; the rest are deterministic placements.
type Positions = Record<string, { x: number; y: number }>
const LAYOUTS: { name: string; run: (g: Graphology) => Positions }[] = [
  { name: 'Circular', run: (g) => circular(g, { scale: 400 }) as Positions },
  { name: 'Circlepack', run: (g) => circlepack(g) as Positions },
  { name: 'Random', run: (g) => random(g, { scale: 800 }) as Positions },
  {
    name: 'Noverlaps',
    run: (g) => noverlap(g, { maxIterations: 60, settings: { margin: 5, ratio: 1 } }) as Positions
  },
  { name: 'Force Directed', run: (g) => forceLayout(g, { maxIterations: 120 }) as Positions },
  {
    name: 'Force Atlas',
    run: (g) => forceAtlas2(g, { iterations: 120, settings: forceAtlas2.inferSettings(g) }) as Positions
  }
]

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
  const selectedNode = useGraphStore.use.selectedNode()
  const setSelectedNode = useGraphStore.use.setSelectedNode()
  const setFocusedNode = useGraphStore.use.setFocusedNode()
  const isFetching = useGraphStore.use.isFetching()
  const graphIsTruncated = useGraphStore.use.graphIsTruncated()
  const hideEncounterEdges = useSettingsStore.use.hideEncounterEdges()
  const showLegend = useSettingsStore.use.showLegend()
  const showPropertyPanel = useSettingsStore.use.showPropertyPanel()
  const showNodeSearchBar = useSettingsStore.use.showNodeSearchBar()
  const colorByCommunity = useSettingsStore.use.colorByCommunity()
  const queryLabel = useSettingsStore.use.queryLabel()

  const wrapperRef = useRef<HTMLDivElement | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)
  const graphRef = useRef<Graph | null>(null)
  const settleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [simRunning, setSimRunning] = useState(true)
  const [layoutMenuOpen, setLayoutMenuOpen] = useState(false)
  // Truncation banner dismissal is keyed to the query label so a new query re-arms it.
  const [dismissedLabel, setDismissedLabel] = useState<string | null>(null)

  // Latest graph for the cosmos click/hover callbacks, which are bound once at
  // Graph creation; the ref keeps them reading current data after a refetch.
  const rawGraphRef = useRef<RawGraph | null>(rawGraph)
  useEffect(() => {
    rawGraphRef.current = rawGraph
  }, [rawGraph])

  const handleFit = useCallback(() => graphRef.current?.fitView(400), [])
  const handleZoomIn = useCallback(() => {
    const g = graphRef.current
    if (g) g.setZoomLevel(g.getZoomLevel() * 1.4, 200)
  }, [])
  const handleZoomOut = useCallback(() => {
    const g = graphRef.current
    if (g) g.setZoomLevel(g.getZoomLevel() / 1.4, 200)
  }, [])

  // LayoutsControl Play/Pause: run the force layout (auto-freezing once it settles)
  // or stop it, mirroring the sigma layout-animation toggle.
  const toggleLayout = useCallback(() => {
    const g = graphRef.current
    if (!g) return
    if (settleTimerRef.current) clearTimeout(settleTimerRef.current)
    setSimRunning((running) => {
      if (running) {
        g.pause()
        return false
      }
      g.start(1)
      settleTimerRef.current = setTimeout(() => {
        g.pause()
        setSimRunning(false)
      }, 4500)
      return true
    })
  }, [])

  // Layout menu: build a graphology graph from the current data, run the chosen
  // layout, and apply the resulting positions to the cosmos points (freezing the
  // simulation so the placement sticks).
  const applyLayout = useCallback((name: string) => {
    const g = graphRef.current
    const raw = rawGraphRef.current
    const spec = LAYOUTS.find((l) => l.name === name)
    if (!g || !raw || raw.nodes.length === 0 || !spec) return

    const gl = new Graphology({ type: 'undirected' })
    const idToIndex = new Map<string, number>()
    raw.nodes.forEach((node, i) => {
      idToIndex.set(node.id, i)
      gl.addNode(node.id, { x: node.x, y: node.y })
    })
    for (const e of raw.edges) {
      if (e.source === e.target || !gl.hasNode(e.source) || !gl.hasNode(e.target)) continue
      if (!gl.hasEdge(e.source, e.target)) gl.addEdge(e.source, e.target)
    }

    const pos = spec.run(gl)
    const out = new Float32Array(raw.nodes.length * 2)
    gl.forEachNode((node) => {
      const idx = idToIndex.get(node)
      const p = pos[node]
      if (idx === undefined || !p) return
      out[idx * 2] = p.x
      out[idx * 2 + 1] = p.y
    })
    if (settleTimerRef.current) clearTimeout(settleTimerRef.current)
    g.pause()
    setSimRunning(false)
    g.setPointPositions(out, true)
    g.fitView(400)
    g.render()
  }, [])

  // Cosmos has no camera rotation, so rotate the settled layout about its centroid.
  const rotate = useCallback((deg: number) => {
    const g = graphRef.current
    if (!g) return
    const pos = g.getPointPositions()
    if (!pos || pos.length === 0) return
    const n = pos.length / 2
    let cx = 0
    let cy = 0
    for (let i = 0; i < n; i++) {
      cx += pos[i * 2]
      cy += pos[i * 2 + 1]
    }
    cx /= n
    cy /= n
    const rad = (deg * Math.PI) / 180
    const cos = Math.cos(rad)
    const sin = Math.sin(rad)
    const out = new Float32Array(pos.length)
    for (let i = 0; i < n; i++) {
      const x = pos[i * 2] - cx
      const y = pos[i * 2 + 1] - cy
      out[i * 2] = cx + x * cos - y * sin
      out[i * 2 + 1] = cy + x * sin + y * cos
    }
    g.setPointPositions(out, true)
    g.render()
  }, [])

  // Generic fullscreen on the viewer wrapper (the sigma FullScreenControl needs a
  // SigmaContainer, so cosmos drives the DOM Fullscreen API directly).
  const toggleFullscreen = useCallback(() => {
    if (document.fullscreenElement) void document.exitFullscreen()
    else void wrapperRef.current?.requestFullscreen()
  }, [])
  useEffect(() => {
    const onChange = () => setIsFullscreen(Boolean(document.fullscreenElement))
    document.addEventListener('fullscreenchange', onChange)
    return () => document.removeEventListener('fullscreenchange', onChange)
  }, [])

  // Mirror the sigma viewer's search behaviour: focus highlights, select pins the
  // node (drives PropertiesView), and the selection zooms the cosmos camera to it.
  const onSearchFocus = useCallback(
    (value: GraphSearchOption | null) => setFocusedNode(value?.type === 'nodes' ? value.id : null),
    [setFocusedNode]
  )
  const onSearchSelect = useCallback(
    (value: GraphSearchOption | null) => setSelectedNode(value?.type === 'nodes' ? value.id : null),
    [setSelectedNode]
  )
  const searchInitSelectedNode = useMemo(
    (): OptionItem | null => (selectedNode ? { type: 'nodes', id: selectedNode } : null),
    [selectedNode]
  )

  useEffect(() => {
    if (!selectedNode) return
    const idx = rawGraph?.nodes.findIndex((n) => n.id === selectedNode) ?? -1
    if (idx >= 0) graphRef.current?.zoomToPointByIndex(idx, 250, 8)
  }, [selectedNode, rawGraph])

  // The truncation notice is informative but the graph is almost always capped at
  // Max Nodes, so auto-dismiss it after a few seconds instead of leaving it pinned.
  useEffect(() => {
    if (!graphIsTruncated || isFetching || dismissedLabel === queryLabel) return
    const timer = setTimeout(() => setDismissedLabel(queryLabel), 6000)
    return () => clearTimeout(timer)
  }, [graphIsTruncated, isFetching, dismissedLabel, queryLabel])

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

    // One RGBA per link — a mid grey with moderate alpha that reads against both
    // the light and dark webui backgrounds (cosmos has no global link colour).
    const linkCount = linkPairs.length / 2
    const linkColors = new Float32Array(linkCount * 4)
    for (let i = 0; i < linkCount; i++) {
      linkColors[i * 4] = 0.55
      linkColors[i * 4 + 1] = 0.6
      linkColors[i * 4 + 2] = 0.66
      linkColors[i * 4 + 3] = 0.5
    }

    return { positions, colors, sizes, links: new Float32Array(linkPairs), linkColors }
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
        // Visible edges: thicker than the default and kept opaque well past the
        // default fade distance so links don't disappear on a spread layout.
        linkWidthScale: 1.4,
        renderLinks: true,
        linkVisibilityDistanceRange: [0, 10000],
        linkVisibilityMinTransparency: 0.4,
        // Stronger gravity + friction so the layout converges and settles instead
        // of drifting/orbiting forever; the layout is also frozen once it settles.
        simulationGravity: 0.25,
        simulationRepulsion: 0.4,
        simulationLinkDistance: 8,
        simulationFriction: 0.92,
        simulationDecay: 3000,
        fitViewOnInit: true,
        enableDrag: true,
        scalePointsOnZoom: true,
        // Ring only the hovered node (a constant per-node outline renders as an
        // offset halo in cosmos, not a tight border, so it is not used).
        renderHoveredPointRing: true,
        hoveredPointRingColor: '#facc15',
        // Match the sigma viewer's interactions: click opens the PropertiesView
        // panel (the real per-entity data); hover focuses the node.
        onClick: (index) => {
          const id =
            index !== undefined ? (rawGraphRef.current?.nodes[index]?.id ?? null) : null
          setSelectedNode(id)
        },
        onPointMouseOver: (index) => {
          setFocusedNode(rawGraphRef.current?.nodes[index]?.id ?? null)
        },
        onPointMouseOut: () => {
          setFocusedNode(null)
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
      g.setLinkColors(buffers.linkColors)
      g.start(1)
      g.render()
      // Let the force layout settle, then freeze it so it stops orbiting. The
      // pause/play control can resume it.
      settleTimerRef.current = setTimeout(() => {
        g.pause()
        setSimRunning(false)
      }, 4500)
    })
    return () => {
      cancelled = true
      if (settleTimerRef.current) clearTimeout(settleTimerRef.current)
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
    <div ref={wrapperRef} className="bg-background relative h-full w-full overflow-hidden">
      <div ref={containerRef} className="h-full w-full" />

      {showPropertyPanel && (
        <div className="absolute top-2 right-2 z-10">
          <PropertiesView />
        </div>
      )}

      <div className="absolute top-2 left-2 flex items-start gap-2">
        <GraphLabels />
        {showNodeSearchBar && (
          <GraphSearch
            value={searchInitSelectedNode}
            onFocus={onSearchFocus}
            onChange={onSearchSelect}
          />
        )}
      </div>

      <div className="bg-background/60 absolute bottom-2 left-2 flex flex-col rounded-xl border-2 backdrop-blur-lg">
        {/* LayoutsControl: run/stop the layout animation, then the layout menu. */}
        <Button
          size="icon"
          variant={controlButtonVariant}
          onClick={toggleLayout}
          tooltip={simRunning ? 'Stop the layout animation' : 'Start the layout animation'}
        >
          {simRunning ? <PauseIcon /> : <PlayIcon />}
        </Button>
        <Popover open={layoutMenuOpen} onOpenChange={setLayoutMenuOpen}>
          <PopoverTrigger asChild>
            <Button
              size="icon"
              variant={controlButtonVariant}
              onClick={() => setLayoutMenuOpen((open) => !open)}
              tooltip="Layout the graph"
            >
              <GripIcon />
            </Button>
          </PopoverTrigger>
          <PopoverContent
            side="right"
            align="start"
            sideOffset={8}
            collisionPadding={5}
            sticky="always"
            className="min-w-auto p-1"
          >
            <Command>
              <CommandList>
                <CommandGroup>
                  {LAYOUTS.map((l) => (
                    <CommandItem
                      key={l.name}
                      onSelect={() => {
                        applyLayout(l.name)
                        setLayoutMenuOpen(false)
                      }}
                      className="cursor-pointer text-xs"
                    >
                      {l.name}
                    </CommandItem>
                  ))}
                </CommandGroup>
              </CommandList>
            </Command>
          </PopoverContent>
        </Popover>

        {/* ZoomControl: rotate, rotate-ccw, reset/fit, zoom in, zoom out. */}
        <Button size="icon" variant={controlButtonVariant} onClick={() => rotate(22.5)} tooltip="Rotate the graph">
          <RotateCwIcon />
        </Button>
        <Button
          size="icon"
          variant={controlButtonVariant}
          onClick={() => rotate(-22.5)}
          tooltip="Rotate the graph counter-clockwise"
        >
          <RotateCcwIcon />
        </Button>
        <Button size="icon" variant={controlButtonVariant} onClick={handleFit} tooltip="Reset zoom">
          <FullscreenIcon />
        </Button>
        <Button size="icon" variant={controlButtonVariant} onClick={handleZoomIn} tooltip="Zoom in">
          <ZoomInIcon />
        </Button>
        <Button size="icon" variant={controlButtonVariant} onClick={handleZoomOut} tooltip="Zoom out">
          <ZoomOutIcon />
        </Button>

        {/* FullScreenControl. */}
        <Button
          size="icon"
          variant={controlButtonVariant}
          onClick={toggleFullscreen}
          tooltip={isFullscreen ? 'Windowed' : 'Full screen'}
        >
          {isFullscreen ? <MinimizeIcon /> : <MaximizeIcon />}
        </Button>
        <LegendButton />
        <Settings />
      </div>

      {showLegend && (
        <div className="absolute right-2 bottom-10 z-0">
          <Legend className="bg-background/60 backdrop-blur-lg" />
        </div>
      )}

      <SettingsDisplay />

      {graphIsTruncated && !isFetching && dismissedLabel !== queryLabel && (
        <div className="absolute top-2 left-1/2 z-20 flex -translate-x-1/2 items-center gap-2 rounded-md bg-amber-500/90 px-3 py-1 text-xs font-medium text-white shadow-md">
          <span>Graph truncated — showing the densest subgraph. Narrow the label or lower Max Nodes for the full view.</span>
          <button
            type="button"
            onClick={() => setDismissedLabel(queryLabel)}
            className="ml-1 font-bold leading-none"
            aria-label="Dismiss"
          >
            ×
          </button>
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
