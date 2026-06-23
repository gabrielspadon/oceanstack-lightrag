import { useEffect, useState } from 'react'
import { DeckGL } from '@deck.gl/react'
import { ScatterplotLayer } from '@deck.gl/layers'
import { Map } from 'react-map-gl/maplibre'
import 'maplibre-gl/dist/maplibre-gl.css'

import { queryMapPorts, queryMapVessels, type MapPort, type MapVessel } from '@/api/lightrag'

/**
 * Geographic view of the maritime data: world ports and recent vessel positions
 * rendered with deck.gl over a MapLibre basemap. The knowledge graph carries no
 * coordinates, so these layers are driven by the /map/* routes that query the
 * oceanstack AIS tables (external.world_ports, derived.vessel_tracks). Tracks,
 * time animation, and graph<->map linked selection are follow-on layers.
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

  useEffect(() => {
    let active = true
    queryMapPorts().then((d) => active && setPorts(d)).catch(() => {})
    queryMapVessels().then((d) => active && setVessels(d)).catch(() => {})
    return () => {
      active = false
    }
  }, [])

  const layers = [
    new ScatterplotLayer<MapVessel>({
      id: 'vessels',
      data: vessels,
      getPosition: (d) => [d.lon, d.lat],
      getFillColor: [80, 180, 255, 160],
      getRadius: 2,
      radiusUnits: 'pixels',
      pickable: true
    }),
    new ScatterplotLayer<MapPort>({
      id: 'ports',
      data: ports,
      getPosition: (d) => [d.lon, d.lat],
      getFillColor: [255, 160, 60, 220],
      getRadius: 4,
      radiusUnits: 'pixels',
      pickable: true
    })
  ]

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
      <div className="bg-background/70 absolute top-2 left-2 z-10 rounded-md px-2 py-1 text-xs backdrop-blur-lg">
        {ports.length.toLocaleString()} ports · {vessels.length.toLocaleString()} recent vessels
      </div>
    </div>
  )
}

export default MapViewer
