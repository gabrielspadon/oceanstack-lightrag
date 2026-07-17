import { beforeAll, describe, expect, test } from 'bun:test'

const storageMock = () => {
  const data = new Map<string, string>()
  return {
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => data.set(key, value),
    removeItem: (key: string) => data.delete(key)
  }
}

beforeAll(() => {
  Object.defineProperty(globalThis, 'localStorage', {
    value: storageMock(),
    configurable: true
  })
})

describe('buildSigmaGraph', () => {
  test('keeps parallel directed assertions and their read metadata', async () => {
    const graphModule = (await import('./useLightragGraph')) as Record<string, unknown>
    expect(typeof graphModule.buildSigmaGraph).toBe('function')

    const { RawGraph } = await import('@/stores/graph')
    const rawGraph = new RawGraph()
    rawGraph.nodes = [
      {
        id: 'source',
        labels: ['Source'],
        properties: { entity_type: 'component' },
        size: 10,
        x: 0,
        y: 0,
        color: '',
        degree: 2
      },
      {
        id: 'target',
        labels: ['Target'],
        properties: { entity_type: 'component' },
        size: 10,
        x: 0,
        y: 0,
        color: '',
        degree: 2
      }
    ]
    rawGraph.edges = [
      {
        id: 'assertion-1',
        source: 'source',
        target: 'target',
        type: 'CALLS',
        properties: {
          assertion_id: 'assertion-1',
          predicate: 'calls',
          direction: 'outbound',
          evidence: { source_path: 'src/a.py', line: 10 },
          traversal: { depth: 1, path: ['source', 'target'] },
          weight: 0.75
        },
        dynamicId: ''
      },
      {
        id: 'assertion-2',
        source: 'source',
        target: 'target',
        type: 'DEPENDS_ON',
        properties: {
          assertion_id: 'assertion-2',
          predicate: 'depends_on',
          direction: 'outbound',
          evidence: { source_path: 'src/b.py', line: 20 },
          traversal: { depth: 2, path: ['source', 'bridge', 'target'] },
          weight: 1
        },
        dynamicId: ''
      }
    ]

    const buildSigmaGraph = graphModule.buildSigmaGraph as (
      graph: InstanceType<typeof RawGraph>
    ) => Promise<{
      type: string
      multi: boolean
      size: number
      source: (edge: string) => string
      target: (edge: string) => string
      getEdgeAttributes: (edge: string) => Record<string, unknown>
    } | null>
    const graph = await buildSigmaGraph(rawGraph)

    expect(graph?.type).toBe('directed')
    expect(graph?.multi).toBe(true)
    expect(graph?.size).toBe(2)
    expect(graph?.source('assertion-1')).toBe('source')
    expect(graph?.target('assertion-1')).toBe('target')
    expect(graph?.getEdgeAttributes('assertion-1')).toMatchObject({
      assertionId: 'assertion-1',
      predicate: 'calls',
      direction: 'outbound',
      evidence: { source_path: 'src/a.py', line: 10 },
      traversal: { depth: 1, path: ['source', 'target'] }
    })
    expect(graph?.getEdgeAttributes('assertion-2')).toMatchObject({
      assertionId: 'assertion-2',
      predicate: 'depends_on',
      direction: 'outbound',
      evidence: { source_path: 'src/b.py', line: 20 },
      traversal: { depth: 2, path: ['source', 'bridge', 'target'] }
    })
  })
})
