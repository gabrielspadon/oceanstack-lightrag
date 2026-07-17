import { create } from 'zustand'

import { GraphPlane } from '@/stores/settings'

/**
 * Generation provenance for one plane, captured from the
 * `X-LightRAG-*` response headers every plane route emits.
 */
export type PlaneProvenance = {
  plane: GraphPlane
  generationId: string
  buildId: string
  sourceRevision: string
  manifestDigest: string
}

interface ProvenanceState {
  byPlane: Partial<Record<GraphPlane, PlaneProvenance>>
  setPlaneProvenance: (provenance: PlaneProvenance) => void
}

/**
 * Latest observed generation identity per plane. Read-only display state:
 * the backend resolves the active generation itself; this store only surfaces
 * what the responses actually came from.
 */
export const useProvenanceStore = create<ProvenanceState>()((set) => ({
  byPlane: {},

  setPlaneProvenance: (provenance) =>
    set((state) => {
      const existing = state.byPlane[provenance.plane]
      if (
        existing &&
        existing.generationId === provenance.generationId &&
        existing.buildId === provenance.buildId &&
        existing.sourceRevision === provenance.sourceRevision &&
        existing.manifestDigest === provenance.manifestDigest
      ) {
        return state
      }
      return { byPlane: { ...state.byPlane, [provenance.plane]: provenance } }
    })
}))

/**
 * Build a PlaneProvenance from response headers, or null when the headers
 * are absent (e.g. error responses or non-plane routes).
 */
export const provenanceFromHeaders = (
  plane: GraphPlane,
  getHeader: (name: string) => string | null | undefined
): PlaneProvenance | null => {
  const generationId = getHeader('x-lightrag-generation-id')
  const buildId = getHeader('x-lightrag-build-id')
  const sourceRevision = getHeader('x-lightrag-source-revision')
  const manifestDigest = getHeader('x-lightrag-manifest-digest')
  if (!generationId || !buildId || !sourceRevision || !manifestDigest) {
    return null
  }
  return { plane, generationId, buildId, sourceRevision, manifestDigest }
}
