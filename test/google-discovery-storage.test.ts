// Protects the model/dimension contract across real storage and query paths.
// Missing DDL, wrong model filtering or a text-SDK query fails these assertions.
// Existing column tests use generic text providers; only fetch and recipe capability are doubled here.
import { afterAll, beforeAll, describe, expect, test } from 'bun:test';
import type { BrainEngine } from '../src/core/engine.ts';
import { PGLiteEngine } from '../src/core/pglite-engine.ts';
import { isolatedPersistencePostgres } from './helpers/persistence-postgres.ts';
import { testBackends } from './helpers/test-backends.ts';
import { installFixtureChunks } from './helpers/page-projection.ts';
import { loadConfigWithEngine } from '../src/core/config.ts';
import { resolveEmbeddingColumn } from '../src/core/search/embedding-column.ts';
import { configureGateway, embedMultimodal, embedQueryMultimodal, resetGateway } from '../src/core/ai/gateway.ts';
import { RECIPES } from '../src/core/ai/recipes/index.ts';
import { google } from '../src/core/ai/recipes/google.ts';
import { hybridSearch } from '../src/core/search/hybrid.ts';

for (const backend of testBackends()) describe(`Google discovery storage (${backend})`, () => {
  let engine: BrainEngine;
  let close: () => Promise<void>;
  const originalFetch = globalThis.fetch;
  const originalGoogle = RECIPES.get('google')!;
  const calls: any[] = [];
  const values = Array(768).fill(0); values[0] = 1;
  beforeAll(async () => {
    if (backend === 'postgres') ({ engine, close } = await isolatedPersistencePostgres(process.env.DATABASE_URL!));
    else { const lite = new PGLiteEngine(); await lite.connect({}); await lite.initSchema(); engine = lite; close = () => lite.disconnect(); }
    RECIPES.set('google', { ...google, touchpoints: { ...google.touchpoints, embedding: {
      ...google.touchpoints.embedding!, supports_multimodal: true, multimodal_models: ['gemini-embedding-2'],
    } } });
    configureGateway({ embedding_model: 'openai:text-embedding-3-large', embedding_dimensions: 1536,
      embedding_multimodal_model: 'google:gemini-embedding-2', env: { GOOGLE_GENERATIVE_AI_API_KEY: 'synthetic' } });
    globalThis.fetch = (async (_url, init) => {
      calls.push(JSON.parse(init!.body as string));
      return Response.json({ embedding: { values }, usageMetadata: { promptTokenCount: 5 } });
    }) as typeof fetch;
  }, 120_000);
  afterAll(async () => { globalThis.fetch = originalFetch; RECIPES.set('google', originalGoogle); resetGateway(); await close?.(); });

  async function target() {
    return resolveEmbeddingColumn({ embeddingColumn: 'embedding_discovery' }, (await loadConfigWithEngine(engine))!);
  }

  test('migration registers the 768d Google column beside the legacy 1024d columns', async () => {
    const column = await target();
    expect(column).toEqual({ name: 'embedding_discovery', type: 'vector', dimensions: 768, embeddingModel: 'google:gemini-embedding-2' });
    const rows = await engine.executeRaw<{ name: string; shape: string }>(`SELECT attname AS name, format_type(atttypid, atttypmod) AS shape FROM pg_attribute
      WHERE attrelid='content_chunks'::regclass AND attname IN ('embedding_discovery','embedding_image','embedding_multimodal') ORDER BY attname`);
    expect(rows).toEqual([{ name: 'embedding_discovery', shape: 'vector(768)' }, { name: 'embedding_image', shape: 'vector(1024)' }, { name: 'embedding_multimodal', shape: 'vector(1024)' }]);
    const indexes = await engine.executeRaw<{ present: boolean }>("SELECT to_regclass('idx_chunks_embedding_discovery') IS NOT NULL AS present");
    expect(indexes[0].present).toBe(true);
  });

  test('explicit descriptor writes and Google queries share model, dimension and modality space', async () => {
    const column = await target();
    await engine.setConfig('search_embedding_column', 'embedding_discovery');
    const [vector] = await embedMultimodal([{ kind: 'content', parts: [{ kind: 'text', text: 'caption' }, { kind: 'text', text: 'context' }] }],
      { embeddingModel: column.embeddingModel, dimensions: column.dimensions });
    for (const [slug, model, modality] of [
      ['resource/current', 'google:gemini-embedding-2', 'image'],
      ['resource/other-model', 'other:model-with-same-dimension', 'image'],
    ] as const) {
      await engine.putPage(slug, { type: 'concept', title: 'Synthetic resource', compiled_truth: 'Synthetic resource content.' });
      await installFixtureChunks(engine, slug, [{ chunk_index: 0, chunk_text: 'Synthetic resource content.', chunk_source: 'compiled_truth',
        embedding: vector, model, modality }]);
    }
    await engine.upsertChunks('resource/current', [{ chunk_index: 0, chunk_text: 'Synthetic resource content.', chunk_source: 'compiled_truth',
      embedding: vector, model: column.embeddingModel, modality: 'image' }], { embeddingColumn: column });
    const query = await embedQueryMultimodal('synthetic resource', { embeddingModel: column.embeddingModel, dimensions: column.dimensions });
    const found = await engine.searchVector(query, { embeddingColumn: column, limit: 10 });
    expect(found.map(row => row.slug)).toEqual(['resource/current']);
    const stored = await engine.executeRaw<{ model: string; dims: number }>(`SELECT cc.model, vector_dims(cc.embedding_discovery) AS dims FROM content_chunks cc JOIN pages p ON p.id=cc.page_id WHERE p.slug='resource/current'`);
    expect(stored).toEqual([{ model: 'google:gemini-embedding-2', dims: 768 }]);
    expect(calls.slice(-2).every(body => body.embedContentConfig.outputDimensionality === 768 && !body.input_type && !body.taskType)).toBe(true);
  });

  test('wrong-width storage is refused without replacing the current vector', async () => {
    const column = await target();
    await expect(engine.upsertChunks('resource/current', [{ chunk_index: 0, chunk_text: 'Synthetic resource content.', chunk_source: 'compiled_truth',
      embedding: new Float32Array(1024).fill(1), model: column.embeddingModel }], { embeddingColumn: column })).rejects.toThrow();
    const [stored] = await engine.executeRaw<{ dims: number }>("SELECT vector_dims(embedding_discovery) AS dims FROM content_chunks cc JOIN pages p ON p.id=cc.page_id WHERE p.slug='resource/current'");
    expect(stored.dims).toBe(768);
  });

  test('hybrid text query uses native Google without cache or a Wiki vector fallback', async () => {
    const column = await target();
    // The wrong-model resource also matches lexical search; inspect provider calls and vector search independently.
    const start = calls.length;
    for (let i = 0; i < 2; i++) await hybridSearch(engine, 'synthetic resource', { embeddingColumn: column, crossModal: 'text', useCache: false, expansion: false });
    expect(calls.length - start).toBe(2);
    expect(calls.slice(start).every(body => body.content.parts[0].text === 'synthetic resource' && body.embedContentConfig.outputDimensionality === 768)).toBe(true);
  });
  test('a rewritten page cannot return the previous projection vector', async () => {
    const column = await target();
    await engine.putPage('resource/current', { type: 'concept', title: 'New resource', compiled_truth: 'A replacement resource body.' });
    const query = new Float32Array(values);
    const found = await engine.searchVector(query, { embeddingColumn: column, limit: 10 });
    expect(found.map(row => row.slug)).not.toContain('resource/current');
  });

});
