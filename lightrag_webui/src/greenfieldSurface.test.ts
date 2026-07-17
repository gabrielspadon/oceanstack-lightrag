import { describe, expect, test } from 'bun:test'

const readSource = async (relativePath: string): Promise<string> =>
  Bun.file(new URL(relativePath, import.meta.url)).text()

describe('greenfield WebUI surface', () => {
  test('offers only the three exact graph planes', async () => {
    const settingsSource = await readSource('./stores/settings.ts')

    expect(settingsSource).toMatch(/'oceanstack_dev'/)
    expect(settingsSource).toMatch(/'oceanstack_product'/)
    expect(settingsSource).toMatch(/'oceanstack_maritime'/)
    expect(settingsSource).toContain('selectedPlane')
  })

  test('has no document-management or graph-mutation components', async () => {
    const removedComponents = [
      './features/DocumentManager.tsx',
      './components/documents/ClearDocumentsDialog.tsx',
      './components/documents/DeleteDocumentsDialog.tsx',
      './components/documents/PipelineStatusDialog.tsx',
      './components/documents/UploadDocumentsDialog.tsx',
      './components/graph/EditablePropertyRow.tsx',
      './components/graph/MergeDialog.tsx',
      './components/graph/PropertyEditDialog.tsx'
    ]

    for (const relativePath of removedComponents) {
      expect(await Bun.file(new URL(relativePath, import.meta.url)).exists()).toBe(false)
    }
  })

  test('contains no unscoped query, graph-read, document, or graph-mutation API paths', async () => {
    const apiSource = await readSource('./api/lightrag.ts')

    for (const prohibitedPath of [
      /'\/query'/,
      /'\/query\/stream'/,
      /'\/graphs/,
      /'\/graph\/label\//,
      /'\/graph\/entity\//,
      /'\/graph\/relation\//,
      /'\/documents/
    ]) {
      expect(apiSource).not.toMatch(prohibitedPath)
    }

    expect(apiSource).toContain('/planes/${plane}/query')
    expect(apiSource).toContain('/planes/${plane}/query/stream')
    expect(apiSource).toContain('/planes/${plane}/graphs')
    expect(apiSource).toContain('/planes/${plane}/graph/label/')
  })

  test('uses MultiDirectedGraph for production and generated graph surfaces', async () => {
    const graphSources = await Promise.all([
      readSource('./hooks/useLightragGraph.tsx'),
      readSource('./hooks/useRandomGraph.tsx'),
      readSource('./stores/graph.ts')
    ])

    for (const source of graphSources) {
      expect(source).toContain('MultiDirectedGraph')
      expect(source).not.toContain('UndirectedGraph')
    }
  })

  test('matches the minimal immutable-generation health contract', async () => {
    const [apiSource, stateSource, statusCardSource, graphLabelsSource] = await Promise.all([
      readSource('./api/lightrag.ts'),
      readSource('./stores/state.ts'),
      readSource('./components/status/StatusCard.tsx'),
      readSource('./components/graph/GraphLabels.tsx')
    ])

    expect(apiSource).toMatch(/status:\s*'ready'/)
    expect(apiSource).toMatch(/generation_runtime:\s*'ready'/)
    expect(apiSource).toMatch(/core_version:\s*string/)
    expect(apiSource).toMatch(/api_version:\s*string/)
    expect(apiSource).toMatch(/webui_available:\s*boolean/)
    expect(stateSource).toContain('health.status === \'ready\'')

    const statusSources = [apiSource, stateSource, statusCardSource, graphLabelsSource]
    for (const source of statusSources) {
      for (const staleSurface of [
        /pipeline_(busy|active|scanning|destructive_busy|pending_enqueues)/,
        /pipelineBusy/,
        /pipelineActive/,
        /LightragQueueStatus/,
        /LightragRoleLLMConfig/,
        /llm_queue_status/,
        /embedding_queue_status/,
        /rerank_queue_status/,
        /storage_workspaces/,
        /doc_status_storage/,
        /parser_routing/,
        /mineru/,
        /docling/,
        /gunicorn/,
        /llm_(binding|model)/,
        /embedding_(binding|model)/,
        /rerank_(binding|model)/
      ]) {
        expect(source).not.toMatch(staleSurface)
      }
    }
  })

  test('does not expose a documents tab visibility key', async () => {
    const visibilitySource = await readSource('./contexts/TabVisibilityProvider.tsx')

    expect(visibilitySource).not.toMatch(/['"]documents['"]\s*:/)
  })
})
