import { useEffect, useMemo, useState } from 'react'
import { DeckGL } from '@deck.gl/react'
import { ScatterplotLayer, PathLayer } from '@deck.gl/layers'
import { Map } from 'react-map-gl/maplibre'
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
 * /map/* routes that query the oceanstack AIS tables. A time slider reveals the
 * recent vessel positions progressively; graph<->map linked selection is the
 * remaining follow-on.
 */

const INITIAL_VIEW_STATE = {
  longitude: -40,
  latitude: 35,
  zoom: 2.2,
  pitch: 0,
  bearing: 0
}

// Free, key-less MapLibre basemap (tiles fetched by the client browser).
const BASEMAP_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json'

const MapViewer = () => {
  const [ports, setPorts] = useState<MapPort[]>([])
  const [vessels, setVessels] = useState<MapVessel[]>([])
  const [tracks, setTracks] = useState<MapTrack[]>([])
  const [showTracks, setShowTracks] = useState(true)
  const [timeCutoff, setTimeCutoff] = useState(1) // 0..1 fraction of the time range; 1 = all

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
    if (vessels.length === 0) return [0, 1]
    let lo = Infinity
    let hi = -Infinity
    for (const v of vessels) {
      if (v.end_time < lo) lo = v.end_time
      if (v.end_time > hi) hi = v.end_time
    }
    return [lo, hi]
  }, [vessels])

  const visibleVessels = useMemo(() => {
    if (timeCutoff >= 1) return vessels
    const cutoff = minTime + (maxTime - minTime) * timeCutoff
    return vessels.filter((v) => v.end_time <= cutoff)
  }, [vessels, timeCutoff, minTime, maxTime])

  const layers = [
    showTracks &&
      new PathLayer<MapTrack>({
        id: 'tracks',
        data: tracks,
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
        <Map mapStyle={BASEMAP_STYLE} />
      </DeckGL>

      <div className="bg-background/70 absolute top-2 left-2 z-10 flex flex-col gap-1 rounded-md px-3 py-2 text-xs backdrop-blur-lg">
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
        <label className="flex items-center gap-2">
          <span>Time</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.02}
            value={timeCutoff}
            onChange={(e) => setTimeCutoff(parseFloat(e.target.value))}
            className="w-32"
          />
        </label>
      </div>
    </div>
  )
}

export default MapViewer
