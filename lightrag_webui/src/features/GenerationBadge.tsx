import { useTranslation } from 'react-i18next'

import { useProvenanceStore } from '@/stores/provenance'
import { useSettingsStore } from '@/stores/settings'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger
} from '@/components/ui/Tooltip'

/**
 * Compact provenance readout for the selected plane: which published
 * generation (build + source revision) the responses are actually served
 * from. Populated from the X-LightRAG-* headers of plane responses, so it
 * appears after the first query or graph fetch on the plane.
 */
const GenerationBadge = () => {
  const { t } = useTranslation()
  const selectedPlane = useSettingsStore.use.selectedPlane()
  const provenance = useProvenanceStore((state) => state.byPlane[selectedPlane])

  if (!provenance) return null

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <span
            className="ml-2 hidden cursor-default self-center rounded-md bg-primary/10 px-2 py-1 font-mono text-xs text-muted-foreground lg:inline"
            data-testid="generation-badge"
          >
            {provenance.buildId}
          </span>
        </TooltipTrigger>
        <TooltipContent side="bottom" className="max-w-96 font-mono text-xs">
          <div>
            {t('header.generation.id', 'Generation')}: {provenance.generationId}
          </div>
          <div>
            {t('header.generation.sourceRevision', 'Source revision')}:{' '}
            {provenance.sourceRevision}
          </div>
          <div>
            {t('header.generation.manifestDigest', 'Manifest digest')}:{' '}
            {provenance.manifestDigest}
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}

export default GenerationBadge
