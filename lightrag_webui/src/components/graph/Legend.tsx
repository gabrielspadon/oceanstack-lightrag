import React from 'react'
import { useTranslation } from 'react-i18next'
import { useGraphStore } from '@/stores/graph'
import { Card } from '@/components/ui/Card'
import { ScrollArea } from '@/components/ui/ScrollArea'

interface LegendProps {
  className?: string
}

// Domain acronyms that read better fully capitalised than Title-cased.
const ACRONYMS = new Set(['ffi', 'gpu', 'sql', 'api', 'id', 'url', 'h3', 'cagg', 'imo', 'mmsi', 'ais'])

/** Turn a raw entity_type (`MaritimeZone`, `ffi_binding`, `shiptype`) into a
 * readable Title-case label, keeping known acronyms upper-case. */
const formatTypeLabel = (type: string): string =>
  type
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/[_-]+/g, ' ')
    .split(/\s+/)
    .filter(Boolean)
    .map((w) => (ACRONYMS.has(w.toLowerCase()) ? w.toUpperCase() : w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()))
    .join(' ')

const Legend: React.FC<LegendProps> = ({ className }) => {
  const { t } = useTranslation()
  const typeColorMap = useGraphStore.use.typeColorMap()

  if (!typeColorMap || typeColorMap.size === 0) {
    return null
  }

  return (
    <Card className={`p-2 max-w-xs ${className}`}>
      <h3 className="text-sm font-medium mb-2">{t('graphPanel.legend')}</h3>
      <ScrollArea className="max-h-80">
        <div className="flex flex-col gap-1">
          {Array.from(typeColorMap.entries()).map(([type, color]) => (
            <div key={type} className="flex items-center gap-2">
              <div
                className="w-4 h-4 rounded-full"
                style={{ backgroundColor: color }}
              />
              <span className="text-xs truncate" title={type}>
                {t(`graphPanel.nodeTypes.${type.toLowerCase().replace(/\s+/g, '')}`, formatTypeLabel(type))}
              </span>
            </div>
          ))}
        </div>
      </ScrollArea>
    </Card>
  )
}

export default Legend
