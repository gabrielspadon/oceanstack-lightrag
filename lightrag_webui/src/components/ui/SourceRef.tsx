/**
 * Shared chunk-provenance row: renders ``source_key @ revision`` with the
 * chunk id in the tooltip. Used by the graph properties panel (typed
 * evidence) and the chat citation list so both surfaces present source
 * references identically.
 */

export type ChunkRef = {
  chunk_id: string
  source_key: string
  source_revision: string
}

const SourceRef = ({ chunkRef }: { chunkRef: ChunkRef }) => (
  <li
    className="truncate font-mono text-xs"
    title={`${chunkRef.chunk_id} @ ${chunkRef.source_revision}`}
  >
    {chunkRef.source_key}
    <span className="opacity-60"> @ {chunkRef.source_revision.slice(0, 12)}</span>
  </li>
)

export default SourceRef
