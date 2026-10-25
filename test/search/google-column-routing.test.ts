// Protects SDK text queries and multimodal column routing through real PGLite retrieval.
// Broad nondefault-column branches lose vector-only hits; previous tests enabled Google's capability.
// Only external fetch is doubled; no production seam is added.
import { afterAll, beforeAll, beforeEach, expect, test } from 'bun:test';
import { PGLiteEngine } from '../../src/core/pglite-engine.ts';
import { configureGateway, resetGateway } from '../../src/core/ai/gateway.ts';
import { hybridSearch } from '../../src/core/search/hybrid.ts';
import type { HybridSearchMeta, ResolvedColumn } from '../../src/core/types.ts';
import { installFixtureChunks } from '../helpers/page-projection.ts';

const originalFetch = globalThis.fetch;
let engine: PGLiteEngine;
const calls: Array<{ url: string; body: any }> = [];
const vector = (dims: number) => { const values = Array(dims).fill(0); values[0] = 1; return values; };
beforeAll(async () => {
  engine = new PGLiteEngine(); await engine.connect({}); await engine.initSchema();
  await engine.executeRaw('ALTER TABLE content_chunks ADD COLUMN embedding_g2 vector(768), ADD COLUMN embedding_voyage vector(1024)');
}, 120_000);
beforeEach(() => { calls.length = 0; });
afterAll(async () => { globalThis.fetch = originalFetch; resetGateway(); await engine.disconnect(); });

async function seed(slug: string, column: ResolvedColumn) {
  await engine.putPage(slug, { type: 'concept', title: 'Synthetic fixture', compiled_truth: 'A synthetic fixture body.' });
  const chunks = [{ chunk_index: 0, chunk_text: 'A synthetic fixture body.', chunk_source: 'compiled_truth' as const,
    model: column.embeddingModel, modality: column.name === 'embedding_image' ? 'image' as const : 'text' as const }];
  if (column.name === 'embedding_image') {
    await installFixtureChunks(engine, slug, chunks.map(c => ({ ...c, embedding_image: new Float32Array(vector(1024)) })));
    return;
  }
  await installFixtureChunks(engine, slug, chunks);
  await engine.upsertChunks(slug, chunks.map(c => ({ ...c, embedding: new Float32Array(vector(column.dimensions)) })), { embeddingColumn: column });
}

test('custom Gemini-2 text column retains SDK vector retrieval with production capability disabled', async () => {
  const column: ResolvedColumn = { name: 'embedding_g2', dimensions: 768, type: 'vector', embeddingModel: 'google:gemini-embedding-2' };
  await seed('fixtures/sdk-text', column);
  configureGateway({ embedding_model: 'openai:text-embedding-3-large', embedding_dimensions: 1536,
    env: { GOOGLE_GENERATIVE_AI_API_KEY: 'synthetic' } });
  globalThis.fetch = (async (url, init) => {
    calls.push({ url: String(url), body: JSON.parse(init!.body as string) });
    return Response.json({ embedding: { values: vector(768) } });
  }) as typeof fetch;
  let meta: HybridSearchMeta | undefined;
  const found = await hybridSearch(engine, 'astral lantern', { embeddingColumn: column, crossModal: 'text', useCache: false,
    expansion: false, onMeta: m => { meta = m; } });
  expect(found.map(r => r.slug)).toContain('fixtures/sdk-text');
  expect(meta?.degraded?.some(d => d.stage === 'embed_unavailable')).toBe(false);
  expect(calls).toHaveLength(1);
  expect(calls[0].url).toEndWith('/models/gemini-embedding-2:embedContent');
  expect(calls[0].body.embedContentConfig).toBeUndefined();
});

for (const unified of [false, true]) test(`custom text column keeps the ${unified ? 'unified' : 'image'} arm in its multimodal column`, async () => {
  const textColumn: ResolvedColumn = { name: 'embedding_voyage', dimensions: 1024, type: 'vector', embeddingModel: 'voyage:voyage-3-large' };
  const mmColumn: ResolvedColumn = { ...textColumn, name: unified ? 'embedding_multimodal' : 'embedding_image', embeddingModel: 'voyage:voyage-multimodal-3' };
  await seed('fixtures/text', textColumn);
  await seed(unified ? 'fixtures/unified' : 'fixtures/image', mmColumn);
  await engine.setConfig('search.unified_multimodal', String(unified));
  await engine.setConfig('search.unified_multimodal_only', String(unified));
  configureGateway({ embedding_model: 'voyage:voyage-3-large', embedding_dimensions: 1024,
    embedding_multimodal_model: 'voyage:voyage-multimodal-3', env: { VOYAGE_API_KEY: 'synthetic' } });
  globalThis.fetch = (async (url, init) => {
    const body = JSON.parse(init!.body as string);
    calls.push({ url: String(url), body });
    return Response.json({ data: [{ embedding: vector(1024) }], usage: { total_tokens: 5 } });
  }) as typeof fetch;
  let meta: HybridSearchMeta | undefined;
  const found = await hybridSearch(engine, 'astral lantern', { embeddingColumn: textColumn, crossModal: 'both',
    useCache: false, expansion: false, onMeta: m => { meta = m; } });
  expect(found.map(r => r.slug)).toContain(unified ? 'fixtures/unified' : 'fixtures/image');
  if (!unified) expect(found.map(r => r.slug)).toContain('fixtures/text');
  expect(meta?.degraded?.some(d => d.stage === 'vector_arm_failed')).toBe(false);
  expect(calls.filter(c => c.url.endsWith('/multimodalembeddings')).map(c => c.body.model)).toEqual(['voyage-multimodal-3']);
  expect(calls.filter(c => c.url.endsWith('/embeddings')).map(c => c.body.model)).toEqual(unified ? [] : ['voyage-3-large']);
});
