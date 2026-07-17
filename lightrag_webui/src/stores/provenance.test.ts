import { describe, expect, test } from 'bun:test'

import { provenanceFromHeaders, useProvenanceStore } from '@/stores/provenance'

const HEADERS: Record<string, string> = {
  'x-lightrag-generation-id': '018f0f7d-c68b-7a2f-8f7d-724a24f9aa01',
  'x-lightrag-build-id': 'build-dev-001',
  'x-lightrag-source-revision': 'source-abc123',
  'x-lightrag-manifest-digest': 'a'.repeat(64)
}

describe('provenanceFromHeaders', () => {
  test('builds provenance from complete headers', () => {
    const provenance = provenanceFromHeaders('oceanstack_dev', (name) => HEADERS[name])
    expect(provenance).toEqual({
      plane: 'oceanstack_dev',
      generationId: HEADERS['x-lightrag-generation-id'],
      buildId: HEADERS['x-lightrag-build-id'],
      sourceRevision: HEADERS['x-lightrag-source-revision'],
      manifestDigest: HEADERS['x-lightrag-manifest-digest']
    })
  })

  test('returns null when any header is missing', () => {
    for (const missing of Object.keys(HEADERS)) {
      const provenance = provenanceFromHeaders('oceanstack_dev', (name) =>
        name === missing ? null : HEADERS[name]
      )
      expect(provenance).toBeNull()
    }
  })
})

describe('useProvenanceStore', () => {
  test('stores provenance per plane and keeps identity stable', () => {
    const provenance = provenanceFromHeaders('oceanstack_product', (name) => HEADERS[name])
    expect(provenance).not.toBeNull()
    if (!provenance) return

    useProvenanceStore.getState().setPlaneProvenance(provenance)
    const first = useProvenanceStore.getState().byPlane
    expect(first.oceanstack_product?.buildId).toBe('build-dev-001')

    // Re-setting identical provenance must not produce a new state object.
    useProvenanceStore.getState().setPlaneProvenance(provenance)
    expect(useProvenanceStore.getState().byPlane).toBe(first)
  })
})
