import type { LightragStatus } from '@/api/lightrag'
import { useTranslation } from 'react-i18next'

const StatusCard = ({ status }: { status: LightragStatus | null }) => {
  const { t } = useTranslation()

  if (!status) {
    return <div className="text-foreground text-xs">{t('graphPanel.statusCard.unavailable')}</div>
  }

  const rows = [
    ['Server', status.status],
    ['Generation runtime', status.generation_runtime],
    ['Core version', status.core_version],
    ['API version', status.api_version],
    ['WebUI assets', status.webui_available ? 'available' : 'unavailable']
  ]

  return (
    <dl className="grid min-w-[300px] grid-cols-[140px_1fr] gap-1 text-xs">
      {rows.map(([label, value]) => (
        <div key={label} className="contents">
          <dt className="text-muted-foreground">{label}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  )
}

export default StatusCard
