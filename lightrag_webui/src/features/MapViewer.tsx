import { useEffect, useMemo, useRef, useState } from 'react'
import { DeckGL, ScatterplotLayer, PathLayer } from 'deck.gl'
import { Map, AttributionControl } from 'react-map-gl/maplibre'
import 'maplibre-gl/dist/maplibre-gl.css'

import {
  queryMapPorts,
  queryMapVessels,
  queryMapTracks,
  type MapPort,
  type MapVessel,
  type MapTrack
} from '@/api/lightrag'
import { useGraphStore } from '@/stores/graph'

/**
 * Geographic view of the maritime data: world ports, recent vessel positions, and
 * recent vessel tracks, rendered with deck.gl over a MapLibre basemap. The
 * knowledge graph carries no coordinates, so these layers are driven by the
 * /map/* routes that query the oceanstack AIS tables. The time slider plays back
 * the recent vessel positions chronologically; graph<->map selection is linked.
 */

const INITIAL_VIEW_STATE = {
  longitude: -40,
  latitude: 35,
  zoom: 2.2,
  pitch: 0,
  bearing: 0
}

// Free, key-less CartoCDN basemaps; the dark/light pair follows the app theme so
// the map does not glare white in dark mode (or stay black in light mode).
const BASEMAP_DARK = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json'
const BASEMAP_LIGHT = 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json'

/** Track the resolved dark/light theme from the `dark` class the ThemeProvider
 * toggles on the document root (covers the explicit and system-driven cases). */
const useIsDark = (): boolean => {
  const [isDark, setIsDark] = useState(() =>
    document.documentElement.classList.contains('dark')
  )
  useEffect(() => {
    const root = document.documentElement
    const sync = () => setIsDark(root.classList.contains('dark'))
    const observer = new MutationObserver(sync)
    observer.observe(root, { attributes: true, attributeFilter: ['class'] })
    sync()
    return () => observer.disconnect()
  }, [])
  return isDark
}

const formatUtc = (epochSeconds: number): string =>
  new Date(epochSeconds * 1000).toISOString().slice(0, 16).replace('T', ' ') + ' UTC'

const MapViewer = () => {
  const isDark = useIsDark()
  const [ports, setPorts] = useState<MapPort[]>([])
  const [vessels, setVessels] = useState<MapVessel[]>([])
  const [tracks, setTracks] = useState<MapTrack[]>([])
  const [showTracks, setShowTracks] = useState(true)
  const [timeCutoff, setTimeCutoff] = useState(1) // 0..1 fraction of the time range; 1 = all
  const [playing, setPlaying] = useState(false)

  // Linked selection: the graph's selected node is a vessel entity_id ("MMSI N");
  // clicking a vessel here sets it, and the matching vessel is highlighted, so the
  // selection follows when switching between the graph and the map.
  const selectedNode = useGraphStore.use.selectedNode()
  const setSelectedNode = useGraphStore.use.setSelectedNode()
  const selectedMmsi = selectedNode?.startsWith('MMSI ') ? selectedNode.slice(5).trim() : null

  useEffect(() => {
    let active = true
    queryMapPorts().then((d) => active && setPorts(d)).catch(() => {})
    queryMapVessels().then((d) => active && setVessels(d)).catch(() => {})
    queryMapTracks().then((d) => active && setTracks(d)).catch(() => {})
    return () => {
      active = false
    }
  }, [])

  const [minTime, maxTime] = useMemo(() => {
    if (vessels.length === 0) return [0, 0]
    let lo = Infinity
    let hi = -Infinity
    for (const v of vessels) {
      if (v.end_time < lo) lo = v.end_time
      if (v.end_time > hi) hi = v.end_time
    }
    return [lo, hi]
  }, [vessels])

  const hasTimeline = maxTime > minTime
  const cutoffEpoch = minTime + (maxTime - minTime) * timeCutoff

  // Playback advances the cutoff across the recent window, then stops at the end.
  const playRef = useRef<ReturnType<typeof setInterval> | null>(null)
  useEffect(() => {
    if (!playing || !hasTimeline) return
    playRef.current = setInterval(() => {
      setTimeCutoff((c) => {
        const next = c + 0.02
        if (next >= 1) {
          setPlaying(false)
          return 1
        }
        return next
      })
    }, 120)
    return () => {
      if (playRef.current) clearInterval(playRef.current)
    }
  }, [playing, hasTimeline])

  const visibleVessels = useMemo(() => {
    if (timeCutoff >= 1 || !hasTimeline) return vessels
    return vessels.filter((v) => v.end_time <= cutoffEpoch)
  }, [vessels, timeCutoff, cutoffEpoch, hasTimeline])

  const layers = [
    showTracks &&
      new PathLayer<MapTrack>({
        id: 'tracks',
        data: tracks,
        // Split segments that cross the 180th meridian so a track does not draw a
        // straight line across the whole map.
        wrapLongitude: true,
        getPath: (d) => [
          [d.start_lon, d.start_lat],
          [d.end_lon, d.end_lat]
        ],
        getColor: [120, 200, 255, 60],
        getWidth: 1,
        widthUnits: 'pixels'
      }),
    new ScatterplotLayer<MapVessel>({
      id: 'vessels',
      data: visibleVessels,
      wrapLongitude: true,
      getPosition: (d) => [d.lon, d.lat],
      getFillColor: [80, 180, 255, 170],
      getRadius: 2,
      radiusUnits: 'pixels',
      pickable: true,
      onClick: (info) => {
        if (info.object) setSelectedNode(`MMSI ${info.object.mmsi}`)
      }
    }),
    new ScatterplotLayer<MapPort>({
      id: 'ports',
      data: ports,
      wrapLongitude: true,
      getPosition: (d) => [d.lon, d.lat],
      getFillColor: [255, 160, 60, 220],
      getRadius: 4,
      radiusUnits: 'pixels',
      pickable: true
    }),
    Boolean(selectedMmsi) &&
      new ScatterplotLayer<MapVessel>({
        id: 'selected-vessel',
        data: vessels.filter((v) => String(v.mmsi) === selectedMmsi),
        wrapLongitude: true,
        getPosition: (d) => [d.lon, d.lat],
        getFillColor: [255, 60, 60, 255],
        getRadius: 7,
        radiusUnits: 'pixels',
        pickable: false
      })
  ].filter(Boolean)

  return (
    <div className="relative h-full w-full">
      <DeckGL
        initialViewState={INITIAL_VIEW_STATE}
        controller={true}
        layers={layers}
        getTooltip={({ object }) =>
          object
            ? 'mmsi' in object
              ? `Vessel ${object.mmsi}`
              : `${object.name}${object.country ? ` (${object.country})` : ''}`
            : null
        }
      >
        {/* Default attribution sits bottom-right and collides with other controls;
            disable it and re-add compact bottom-left where nothing else renders. */}
        <Map mapStyle={isDark ? BASEMAP_DARK : BASEMAP_LIGHT} attributionControl={false}>
          <AttributionControl position="bottom-left" compact />
        </Map>
      </DeckGL>

      <div className="bg-background/70 text-foreground absolute top-2 left-2 z-10 flex flex-col gap-1 rounded-md px-3 py-2 text-xs backdrop-blur-lg">
        <div>
          {ports.length.toLocaleString()} ports · {visibleVessels.length.toLocaleString()} vessels ·{' '}
          {tracks.length.toLocaleString()} tracks
        </div>
        <label className="flex items-center gap-1">
          <input
            type="checkbox"
            checked={showTracks}
            onChange={(e) => setShowTracks(e.target.checked)}
          />
          Tracks
        </label>
        {hasTimeline && (
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => {
                  // Restart from the beginning if pressing play at the end.
                  if (!playing && timeCutoff >= 1) setTimeCutoff(0)
                  setPlaying((p) => !p)
                }}
                className="bg-background/80 rounded border px-2 py-0.5"
              >
                {playing ? 'Pause' : 'Play'}
              </button>
              <span className="tabular-nums">{formatUtc(cutoffEpoch)}</span>
            </div>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={timeCutoff}
              onChange={(e) => {
                setPlaying(false)
                setTimeCutoff(parseFloat(e.target.value))
              }}
              className="w-44"
            />
          </div>
        )}
      </div>
    </div>
  )
}

export default MapViewer
