// Protects persisted 1024d legacy writes and registered image-query model/dimension identity.
// Dropping the dimension override or passing only a column name loses the expected rows.
// Existing coverage used fixed-width Voyage; only the external HTTP response is doubled here.
import { afterAll, afterEach, beforeAll, beforeEach, expect, test } from 'bun:test';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { PGLiteEngine } from '../src/core/pglite-engine.ts';
import { configureGateway, embedMultimodal, embedMultimodalSafe, resetGateway } from '../src/core/ai/gateway.ts';
import { importImageFile } from '../src/core/import-file.ts';
import { runReindexMultimodal } from '../src/commands/reindex-multimodal.ts';
import { searchByImage } from '../src/core/search/by-image.ts';
import { operations } from '../src/core/operations.ts';
import type { ResolvedColumn } from '../src/core/types.ts';
import { installFixtureChunks } from './helpers/page-projection.ts';
import { resetPgliteState } from './helpers/reset-pglite.ts';
import { withEnv } from './helpers/with-env.ts';

const originalFetch = globalThis.fetch;
const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aV9kAAAAASUVORK5CYII=', 'base64');
const values = (dims: number) => [1, ...Array(dims - 1).fill(0)];
let engine: PGLiteEngine;
let root: string;
let calls: Array<{ url: string; body: any }>;
beforeAll(async () => {
  engine = new PGLiteEngine(); await engine.connect({}); await engine.initSchema();
  await engine.executeRaw('ALTER TABLE content_chunks ALTER COLUMN embedding_discovery TYPE vector(129)');
  root = mkdtempSync(join(tmpdir(), 'gbrain-mm-boundaries-'));
  writeFileSync(join(root, 'pixel.png'), png);
}, 120000);
beforeEach(async () => {
  await resetPgliteState(engine);
  calls = [];
  configureGateway({ embedding_model: 'openai:text-embedding-3-small', embedding_dimensions: 768,
    embedding_multimodal_model: 'litellm:global-mm', base_urls: { litellm: 'http://127.0.0.1:9/proxy' }, env: {} });
  globalThis.fetch = (async (url, init) => {
    const body = JSON.parse(init!.body as string); calls.push({ url: String(url), body });
    return Response.json({ data: [{ embedding: values(body.model === 'registered-mm' ? 129 : 1024) }] });
  }) as typeof fetch;
});
afterEach(() => { globalThis.fetch = originalFetch; resetGateway(); });
afterAll(async () => { await engine.disconnect(); if (root) rmSync(root, { recursive: true, force: true }); });

test('image import stores the legacy 1024d vector when primary and proxy defaults differ', async () => {
  const result = await importImageFile(engine, join(root, 'pixel.png'), 'fixtures/pixel.png');
  expect(result.status).toBe('imported');
  const rows = await engine.executeRaw('SELECT vector_dims(embedding_image) AS dims FROM content_chunks WHERE modality = \'image\'');
  expect(rows).toEqual([{ dims: 1024 }]);
  expect(calls).toHaveLength(1);
  expect(calls[0].url).toBe('http://127.0.0.1:9/proxy/embeddings');
});

test('multimodal reindex stores the legacy 1024d vector instead of inheriting primary width', async () => {
  await engine.putPage('fixtures/reindex', { type: 'concept', title: 'Synthetic fixture', compiled_truth: 'Synthetic reindex body.' });
  await installFixtureChunks(engine, 'fixtures/reindex', [{ chunk_index: 0, chunk_text: 'Synthetic reindex body.', chunk_source: 'compiled_truth' }]);
  const result = await withEnv({ GBRAIN_HOME: root }, () => runReindexMultimodal(engine, { yes: true }));
  expect(result.reembedded).toBe(1);
  expect(result.failed).toBe(0);
  expect(result.pending_after).toBe(0);
  expect(await engine.executeRaw('SELECT vector_dims(embedding_multimodal) AS dims FROM content_chunks')).toEqual([{ dims: 1024 }]);
});

async function registeredImages() {
  const column: ResolvedColumn = { name: 'embedding_discovery', type: 'vector', dimensions: 129, embeddingModel: 'litellm:registered-mm' };
  await engine.setConfig('embedding_columns', JSON.stringify({ embedding_discovery: { type: 'vector', dimensions: 129, provider: 'litellm:registered-mm' } }));
  for (const [slug, model] of [['fixtures/current', 'litellm:registered-mm'], ['fixtures/other-space', 'litellm:other-mm']]) {
    await engine.putPage(slug, { type: 'concept', title: 'Synthetic fixture', compiled_truth: 'Synthetic image resource.' });
    const chunks = [{ chunk_index: 0, chunk_text: 'Synthetic image resource.', chunk_source: 'compiled_truth' as const, model, modality: 'image' as const }];
    await installFixtureChunks(engine, slug, chunks);
    await engine.upsertChunks(slug, chunks.map(c => ({ ...c, embedding: new Float32Array(values(129)) })), { embeddingColumn: { ...column, embeddingModel: model } });
  }
}

test('searchByImage retrieves only the registered vector space with a nondefault width and refinement', async () => {
  await registeredImages();
  const found = await searchByImage(engine, { base64: png.toString('base64'), mime: 'image/png' }, { embeddingColumn: 'embedding_discovery', query: 'synthetic refinement' });
  expect(found.map(r => r.slug)).toEqual(['fixtures/current']);
  expect(calls.map(c => c.body.model)).toEqual(['registered-mm', 'registered-mm']);
  expect(calls[1].body.input[0]).toEqual({ type: 'input_text', text: 'synthetic refinement' });
});

test('query image operation retrieves only the registered vector space with a nondefault width', async () => {
  await registeredImages();
  const query = operations.find(o => o.name === 'query')!;
  const found = await query.handler({ engine, remote: false, config: null, logger: console, dryRun: false } as any,
    { image: png.toString('base64'), image_mime: 'image/png', embedding_column: 'embedding_discovery' }) as Array<{ slug: string }>;
  expect(found.map(r => r.slug)).toEqual(['fixtures/current']);
  expect(calls.map(c => c.body.model)).toEqual(['registered-mm']);
});

test('Voyage refuses an incompatible requested width before provider transmission', async () => {
  configureGateway({ embedding_model: 'voyage:voyage-multimodal-3', env: { VOYAGE_API_KEY: 'synthetic' } });
  for (const dimensions of [768, 129]) {
    await expect(embedMultimodal([{ kind: 'text', text: 'synthetic query' }], { dimensions })).rejects.toThrow('1024');
  }
  expect(calls).toEqual([]);
  const [vector] = await embedMultimodal([{ kind: 'text', text: 'synthetic query' }], { dimensions: 1024 });
  expect(vector.length).toBe(1024);
});

test('Safe refuses missing and extra provider vectors instead of reporting an incomplete success', async () => {
  configureGateway({ embedding_model: 'voyage:voyage-multimodal-3', env: { VOYAGE_API_KEY: 'synthetic' } });
  for (const count of [0, 2]) {
    globalThis.fetch = (async () => Response.json({ data: Array.from({ length: count }, () => ({ embedding: values(1024) })) })) as unknown as typeof fetch;
    const result = await embedMultimodalSafe([{ kind: 'text', text: 'synthetic query' }]);
    expect(result.failedIndices).toEqual([0]);
    expect(result.embeddings).toEqual([undefined]);
  }
});
