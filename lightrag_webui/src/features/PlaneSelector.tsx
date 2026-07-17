import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/Select'
import { useGraphStore } from '@/stores/graph'
import {
  GRAPH_PLANES,
  type GraphPlane,
  useSettingsStore
} from '@/stores/settings'
import { SearchHistoryManager } from '@/utils/SearchHistoryManager'

const PlaneSelector = () => {
  const selectedPlane = useSettingsStore.use.selectedPlane()

  const handlePlaneChange = (plane: GraphPlane) => {
    if (plane === selectedPlane) return

    const settings = useSettingsStore.getState()
    settings.setSelectedPlane(plane)
    settings.setQueryLabel('*')
    settings.setRetrievalHistory([])
    settings.triggerSearchLabelDropdownRefresh()
    SearchHistoryManager.clearHistory()

    const graph = useGraphStore.getState()
    graph.reset()
    graph.setGraphDataFetchAttempted(false)
    graph.setLabelsFetchAttempted(false)
    graph.incrementGraphDataVersion()
  }

  return (
    <Select value={selectedPlane} onValueChange={(value) => handlePlaneChange(value as GraphPlane)}>
      <SelectTrigger className="h-8 w-[220px]" aria-label="Graph plane">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {GRAPH_PLANES.map((plane) => (
          <SelectItem key={plane} value={plane}>
            {plane}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

export default PlaneSelector
