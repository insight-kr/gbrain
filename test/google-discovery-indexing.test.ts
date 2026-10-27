// Protects provider identity and provenance when the primary and Discovery widths coincide.
// OpenAI returns e0; native Google returns e1. Only e1 may reach Discovery.
import { afterAll, beforeAll, beforeEach, describe, expect, test } from 'bun:test';
import type { BrainEngine } from '../src/core/engine.ts';
import { PGLiteEngine } from '../src/core/pglite-engine.ts';
import { configureGateway, resetGateway, __setEmbedTransportForTests } from '../src/core/ai/gateway.ts';
import { importFromContent, importCodeFile } from '../src/core/import-file.ts';
import { runEmbedCore } from '../src/commands/embed.ts';
import { embedStaleForSource } from '../src/core/embed-stale.ts';
import { reembedPageWithContextualRetrieval } from '../src/core/contextual-retrieval-service.ts';
import { stampEmbeddingInputs } from '../src/core/page-state/projections.ts';
import { createHash, randomUUID } from 'node:crypto';
import { segmentPage } from './helpers/segmented-chunks-fixture.ts';
import { isolatedPersistencePostgres } from './helpers/persistence-postgres.ts';
import { testBackends } from './helpers/test-backends.ts';

import { admitWrite, claimNextWrite } from '../src/core/persistence/journal.ts';
import { publishMutation } from '../src/core/persistence/coordinator.ts';
import { localHostId, registerLocalWriter } from '../src/core/persistence/identity.ts';
import { submissionAuthority } from '../src/core/persistence/authority.ts';
import { runPersistenceEffects } from '../src/core/persistence/effects.ts';
import { installPageProjection, readProjectionSnapshot } from '../src/core/page-state/projections.ts';
import type { OperationContext } from '../src/core/ops/contract.ts';

const MODEL = 'google:gemini-embedding-2';
const unit = (i: number) => Array.from({ length: 1536 }, (_, j) => Number(i === j));
for (const backend of testBackends()) describe(`Discovery text indexing (${backend})`, () => {
  let engine: BrainEngine;
  let close: () => Promise<void>;
  let openaiCalls = 0;
  const bodies: any[] = [];
  const originalFetch = globalThis.fetch;
  beforeAll(async () => {
    configureGateway({ embedding_model: 'openai:text-embedding-3-large', embedding_dimensions: 1536,
      embedding_multimodal_model: MODEL, env: { OPENAI_API_KEY: 'sk-test-fake', GOOGLE_GENERATIVE_AI_API_KEY: 'synthetic' } });
    __setEmbedTransportForTests((async ({ values }: { values: string[] }) => {
      openaiCalls++;
      return { embeddings: values.map(() => unit(0)), usage: { tokens: 1 } };
    }) as never);
    globalThis.fetch = (async (_url, init) => {
      bodies.push({ ...JSON.parse(init!.body as string), url: String(_url) });
      return Response.json({ embedding: { values: unit(1) }, usageMetadata: { promptTokenCount: 1 } });
    }) as typeof fetch;
    if (backend === 'postgres') ({ engine, close } = await isolatedPersistencePostgres(process.env.DATABASE_URL!));
    else { engine = new PGLiteEngine(); await engine.connect({}); await engine.initSchema(); close = () => engine.disconnect(); }
    await engine.setConfig('search_embedding_column', 'embedding_discovery');
  }, 120_000);
  beforeEach(async () => { await engine.executeRaw('DELETE FROM pages'); openaiCalls = 0; bodies.length = 0; });
  afterAll(async () => { globalThis.fetch = originalFetch; __setEmbedTransportForTests(null); resetGateway(); await close?.(); });
  const seed = (noEmbed = true) => importFromContent(engine, 'notes/discovery',
    '---\ntitle: Discovery fixture\ntype: note\n---\n\nA synthetic text resource for Discovery indexing.\n', { noEmbed });
  const stored = () => engine.executeRaw<{ model: string; dims: number; first: number; second: number; hash: string; chunk_text: string; chunk_source: string }>(
    `SELECT model,vector_dims(embedding_discovery) AS dims,(embedding_discovery::real[])[1] AS first,
      (embedding_discovery::real[])[2] AS second,embedding_input_hash AS hash,chunk_text,chunk_source FROM content_chunks`);
  async function assertGoogle() {
    const rows = await stored();
    expect(rows.length).toBeGreaterThan(0);
    for (const row of rows) expect(row).toMatchObject({ model: MODEL, dims: 1536, first: 0, second: 1 });
    expect(openaiCalls).toBe(0);
    expect(bodies.length).toBeGreaterThan(0);
    expect(bodies.every(body => body.embedContentConfig.outputDimensionality === 1536 && body.url.endsWith('/models/gemini-embedding-2:embedContent'))).toBe(true);
    const signatures = await engine.executeRaw<{ signature: string | null }>('SELECT embedding_signature AS signature FROM pages');
    for (const row of signatures) if (row.signature) expect(row.signature).toBe('google:gemini-embedding-2:1536');
    return rows;
  }
  test('import embeds with the descriptor model and records the actual input space', async () => {
    await seed(false);
    const rows = await assertGoogle();
    const hash = (text: string) => createHash('sha256').update(text).digest('hex');
    for (const row of rows) expect(row.hash).toBe(hash(JSON.stringify(['embedding-input-v1', 'embedding_discovery', MODEL, 1536,
      'title', null, hash(`<context>Discovery fixture\n</context>\n${row.chunk_text}`)])));
  });
  for (const mode of ['stale', 'slug', 'all', 'backfill'] as const) test(`${mode} uses Google with a primary OpenAI model of the same width`, async () => {
    await seed();
    if (mode === 'backfill') expect((await embedStaleForSource(engine, 'default')).embedded).toBe(1);
    else {
      const result = await runEmbedCore(engine, { ...(mode === 'slug' ? { slugs: ['notes/discovery'] } : { [mode]: true }), dryRun: false, quiet: true });
      expect(result).toMatchObject({ embedded: 1, failures: 0 });
    }
    await assertGoogle();
  });
  for (const mode of ['stale', 'slug', 'all', 'backfill'] as const) test(`segmented ${mode} uses the same descriptor model`, async () => {
    await segmentPage(engine, 'notes/discovery', { generation: 1, occurrences: [{ id: 1, byteStart: 0, genFrom: 1,
      chunks: [{ text: 'A synthetic segmented resource for Discovery.' }] }] });
    if (mode === 'backfill') expect((await embedStaleForSource(engine, 'default')).embedded).toBe(1);
    else expect(await runEmbedCore(engine, { ...(mode === 'slug' ? { slugs: ['notes/discovery'] } : { [mode]: true }), quiet: true })).toMatchObject({ embedded: 1, failures: 0 });
    await assertGoogle();
  });
  test('contextual retrieval keeps its text wrapper in the descriptor space', async () => {
    await seed();
    await engine.setConfig('contextual_retrieval.mode', 'title');
    try {
      const result = await reembedPageWithContextualRetrieval({ engine, pageSlug: 'notes/discovery', sourceId: 'default', globalMode: 'title' });
      expect(result.kind).toBe('success');
      await assertGoogle();
      expect(bodies.some(body => body.content.parts[0].text.includes('Discovery fixture'))).toBe(true);
    } finally { await engine.setConfig('contextual_retrieval.mode', 'none'); }
  });
  for (const segmented of [false, true]) test(`contextual synopsis uses the descriptor (${segmented ? 'segmented' : 'inline'})`, async () => {
    if (segmented) await segmentPage(engine, 'notes/discovery', { generation: 1, occurrences: [{ id: 1, byteStart: 0, genFrom: 1,
      chunks: [{ text: 'A synthetic segmented synopsis resource.' }] }] });
    else await seed();
    const result = await reembedPageWithContextualRetrieval({ engine, pageSlug: 'notes/discovery', sourceId: 'default', globalMode: 'per_chunk_synopsis',
      generateSynopsis: async () => ({ kind: 'success', synopsis: 'Synthetic synopsis.' }) });
    expect(result).toMatchObject({ kind: 'success', mode_applied: 'per_chunk_synopsis', chunks_embedded: 1 });
    await assertGoogle();
    expect(bodies.some(body => body.content.parts[0].text.includes('Synthetic synopsis.'))).toBe(true);
  });
  test('code import embeds new chunks in the descriptor space', async () => {
    await importCodeFile(engine, 'example.ts', 'export function syntheticResource() { return "Discovery"; }');
    await assertGoogle();
  });
  test('code import replaces a reusable chunk from a different model', async () => {
    const code = 'export function syntheticResource() { return "Discovery"; }';
    await importCodeFile(engine, 'example.ts', code);
    await engine.executeRaw("UPDATE content_chunks SET model='openai:text-embedding-3-large', embedding_discovery=$1::vector", [JSON.stringify(unit(0))]);
    openaiCalls = 0; bodies.length = 0;
    await importCodeFile(engine, 'example.ts', code, { force: true });
    await assertGoogle();
  });
  test('post-commit import embeds with the captured descriptor', async () => {
    let complete: (() => Promise<unknown>) | undefined;
    await importFromContent(engine, 'notes/discovery', '---\ntitle: Deferred fixture\ntype: note\n---\n\nA deferred synthetic resource.',
      { onPostCommitEmbedding: run => { complete = run; } });
    expect(bodies.length).toBe(0);
    expect(complete).toBeDefined();
    await complete!();
    await assertGoogle();
  });
  test('Discovery writes do not require the primary key or primary output width', async () => {
    __setEmbedTransportForTests(null);
    configureGateway({ embedding_model: 'openai:text-embedding-3-large', embedding_dimensions: 1024,
      embedding_multimodal_model: MODEL, env: { GOOGLE_GENERATIVE_AI_API_KEY: 'synthetic' } });
    try {
      await seed();
      expect(await runEmbedCore(engine, { stale: true, quiet: true })).toMatchObject({ embedded: 1, failures: 0 });
      await assertGoogle();
    } finally {
      configureGateway({ embedding_model: 'openai:text-embedding-3-large', embedding_dimensions: 1536,
        embedding_multimodal_model: MODEL, env: { OPENAI_API_KEY: 'sk-test-fake', GOOGLE_GENERATIVE_AI_API_KEY: 'synthetic' } });
      __setEmbedTransportForTests((async ({ values }: { values: string[] }) => {
        openaiCalls++; return { embeddings: values.map(() => unit(0)), usage: { tokens: 1 } };
      }) as never);
    }
  });
  for (const segmented of [false, true]) test(`persistence effect uses the descriptor (${segmented ? 'segmented' : 'inline'})`, async () => {
    await registerLocalWriter(engine, 'cli');
    const sourceId = `discovery-${randomUUID()}`;
    const [source] = await engine.executeRaw<{ incarnation: string }>('INSERT INTO sources(id,name) VALUES($1,$1) RETURNING incarnation', [sourceId]);
    const ctx: OperationContext = { engine, config: { engine: engine.kind }, remote: false, dryRun: false, sourceId,
      logger: { info() {}, warn() {}, error() {} } };
    const authority = await submissionAuthority(ctx, 'put_page', sourceId, source!.incarnation, 'notes/discovery');
    const admitted = await admitWrite(engine, { principal: authority.principal, authority, operation: 'put_page', sourceId,
      sourceIncarnation: source!.incarnation, slug: 'notes/discovery', requestId: randomUUID(), callerIntent: { body: 'Synthetic resource.' }, intent: { body: 'Synthetic resource.' } });
    const row = (await claimNextWrite(engine, localHostId()))!;
    expect(row.id).toBe(admitted.id);
    await publishMutation(engine, row, { observedRevision: null, apply: async tx => {
      if (segmented) await segmentPage(tx, 'notes/discovery', { sourceId, generation: 1, occurrences: [{ id: 1, byteStart: 0, genFrom: 1, chunks: [{ text: 'Synthetic resource.' }] }] });
      else await tx.putPage('notes/discovery', { type: 'note', title: 'Effect fixture', compiled_truth: 'Synthetic resource.' }, { sourceId });
      return {};
    } }, localHostId());
    if (!segmented) {
      const prepared = (await readProjectionSnapshot(engine, 'notes/discovery', sourceId, { allowUnsealed: true }))!;
      await installPageProjection(engine, prepared, [{ chunk_index: 0, chunk_source: 'compiled_truth', chunk_text: 'Synthetic resource.' }], { seal: true });
    }
    await engine.executeRaw("UPDATE persistence_effects SET next_attempt_at=now()+interval '1 hour'");
    await engine.executeRaw("UPDATE persistence_effects SET next_attempt_at=now() WHERE request_id=$1::uuid AND kind='embedding'", [row.id]);
    await runPersistenceEffects(engine, { engine: engine.kind }, { hostId: localHostId(), limit: 1 });
    await assertGoogle();
    const [effect] = await engine.executeRaw<{ state: string }>("SELECT state FROM persistence_effects WHERE request_id=$1::uuid AND kind='embedding'", [row.id]);
    expect(effect!.state).toBe('committed');
  });
  test('an import cannot certify a vector labelled with another actual model as Google', async () => {
    await expect(stampEmbeddingInputs(engine, [{ chunk_index: 0, chunk_text: 'Wrong-space vector.', chunk_source: 'compiled_truth',
      embedding: new Float32Array(unit(0)), model: 'openai:text-embedding-3-large' }], null,
    { title: 'Wrong-space fixture', tier: 'none', corpusGeneration: null })).rejects.toMatchObject({ code: 'discovery_schema_mismatch' });
    expect(openaiCalls).toBe(0);
    expect(bodies.length).toBe(0);
    expect(await stored()).toEqual([]);
  });
});
