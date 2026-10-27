// Protects provider identity and provenance when the primary and Discovery widths coincide.
// OpenAI returns e0; native Google returns e1. Only e1 may reach Discovery.
// Regressions: omitted dispatch overrides, false provenance, mismatched effect models and
// primary signatures used for Discovery completion/costs. Existing coverage lacked these
// recovery and reporting paths. Real engines and provider-boundary doubles need no new seam.
import { afterAll, beforeAll, beforeEach, describe, expect, test } from 'bun:test';
import type { BrainEngine } from '../src/core/engine.ts';
import { PGLiteEngine } from '../src/core/pglite-engine.ts';
import { configureGateway, resetGateway, __setEmbedTransportForTests } from '../src/core/ai/gateway.ts';
import { importFromContent, importCodeFile } from '../src/core/import-file.ts';
import { runEmbedCore } from '../src/commands/embed.ts';
import { embedStalePages, embedStaleForSource } from '../src/core/embed-stale.ts';
import { reembedPageWithContextualRetrieval } from '../src/core/contextual-retrieval-service.ts';
import { reindexCodeProjection } from '../src/core/persistence/projection-reindex.ts';
import { disposePersistenceConsumer } from '../src/core/persistence/service.ts';
import { currentEmbeddingSignature } from '../src/core/embedding.ts';
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
import { installPageEmbeddings, installPageProjection, readProjectionSnapshot } from '../src/core/page-state/projections.ts';
import { settleNoopEmbeddingEffects } from '../src/core/persistence/embedding-noop-settle.ts';
import { readEmbedBackfillBlockers } from '../src/core/embed-backfill-stall.ts';
import { holdPageEmbedding } from '../src/core/page-embedding-budget.ts';
import { budgetLimitedReport, readBudgetLimited, noteBudgetLimited } from '../src/core/held-embedding.ts';
import { partialEmbeddingEntry } from '../src/commands/doctor/checks/content-quality.ts';
import type { DoctorContext } from '../src/commands/doctor/context.ts';
import { installFixtureChunks } from './helpers/page-projection.ts';
import { runInlineCostGate } from '../src/core/sync-cost-gate.ts';
import { runOversizedBackfill } from '../src/core/oversized-backfill.ts';
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
  for (const managed of [false, true]) for (const dimensions of [768, 1024, 1536]) {
    test(`reindex-code uses Google before stamping provenance (managed=${managed}, primary=${dimensions})`, async () => {
      configureGateway({ embedding_model: 'openai:text-embedding-3-large', embedding_dimensions: dimensions,
        embedding_multimodal_model: MODEL, env: { OPENAI_API_KEY: 'sk-test-fake', GOOGLE_GENERATIVE_AI_API_KEY: 'synthetic' } });
      __setEmbedTransportForTests((async ({ values }: { values: string[] }) => {
        openaiCalls++;
        return { embeddings: values.map(() => Array.from({ length: dimensions }, (_, i) => Number(i === 0))), usage: { tokens: 1 } };
      }) as never);
      const { slug } = await importCodeFile(engine, 'reindexed.ts', 'export function syntheticCode() { return 2; }', { noEmbed: true });
      await engine.executeRaw('UPDATE persistence_brain SET enabled=$1', [managed]);
      try {
        expect(await reindexCodeProjection(engine, slug, 'default', { force: true })).toMatchObject({ status: 'imported' });
        const rows = await assertGoogle();
        const hash = (text: string) => createHash('sha256').update(text).digest('hex');
        for (const row of rows) expect(row.hash).toBe(hash(JSON.stringify(['embedding-input-v1', 'embedding_discovery', MODEL, 1536,
          'none', null, hash(row.chunk_text)])));
        const calls = bodies.length;
        expect(await runEmbedCore(engine, { stale: true, quiet: true })).toMatchObject({ embedded: 0, failures: 0 });
        expect(bodies.length).toBe(calls);
        await assertGoogle();
      } finally {
        await disposePersistenceConsumer(engine);
        await engine.executeRaw('UPDATE persistence_brain SET enabled=false');
        configureGateway({ embedding_model: 'openai:text-embedding-3-large', embedding_dimensions: 1536,
          embedding_multimodal_model: MODEL, env: { OPENAI_API_KEY: 'sk-test-fake', GOOGLE_GENERATIVE_AI_API_KEY: 'synthetic' } });
        __setEmbedTransportForTests((async ({ values }: { values: string[] }) => {
          openaiCalls++; return { embeddings: values.map(() => unit(0)), usage: { tokens: 1 } };
        }) as never);
      }
    }, 120_000);
  }
  for (const managed of [false, true]) test(`reindex-code refuses an unavailable Discovery provider before any call (managed=${managed})`, async () => {
    const { slug } = await importCodeFile(engine, 'unavailable.ts', 'export const synthetic = 3;', { noEmbed: true });
    configureGateway({ embedding_model: 'openai:text-embedding-3-large', embedding_dimensions: 1536,
      embedding_multimodal_model: MODEL, env: { OPENAI_API_KEY: 'sk-test-fake' } });
    await engine.executeRaw('UPDATE persistence_brain SET enabled=$1', [managed]);
    try {
      await expect(reindexCodeProjection(engine, slug, 'default', { force: true })).rejects.toThrow();
      expect(openaiCalls).toBe(0);
      expect(bodies.length).toBe(0);
      expect((await stored()).every(chunk => chunk.dims === null && chunk.hash === null)).toBe(true);
    } finally {
      await disposePersistenceConsumer(engine);
      await engine.executeRaw('UPDATE persistence_brain SET enabled=false');
      configureGateway({ embedding_model: 'openai:text-embedding-3-large', embedding_dimensions: 1536,
        embedding_multimodal_model: MODEL, env: { OPENAI_API_KEY: 'sk-test-fake', GOOGLE_GENERATIVE_AI_API_KEY: 'synthetic' } });
    }
  }, 120_000);
  test('cycle and repair stale pages use the descriptor with a primary signature argument', async () => {
    await seed();
    expect(await embedStalePages(engine, ['notes/discovery'], 'default', { embeddingSignature: currentEmbeddingSignature()! }))
      .toMatchObject({ embedded: 1, pagesProcessed: 1, aborted: false });
    await assertGoogle();
    expect(await engine.executeRaw("SELECT embedding_signature FROM pages WHERE source_id='default' AND slug='notes/discovery'"))
      .toEqual([{ embedding_signature: 'google:gemini-embedding-2:1536' }]);
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
  for (const segmented of [false, true]) for (const mode of ['embed', 'mismatch', 'settle'] as const) test(`persistence effect ${mode} (${segmented ? 'segmented' : 'inline'})`, async () => {
    const mismatched = mode === 'mismatch';
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
    if (mode === 'settle') {
      await runEmbedCore(engine, { stale: true, quiet: true });
      await assertGoogle();
      expect((await settleNoopEmbeddingEffects(engine, localHostId(), currentEmbeddingSignature()!)).settled).toBe(1);
      return;
    }
    let explicitCalls = 0;
    await runPersistenceEffects(engine, { engine: engine.kind }, { hostId: localHostId(), limit: 1,
      ...(mismatched && { embedding: { model: 'openai:text-embedding-3-large', signature: 'openai:text-embedding-3-large:1536',
        embed: async (texts: string[]) => { explicitCalls++; return texts.map(() => new Float32Array(unit(0))); } } }) });
    const [effect] = await engine.executeRaw<{ state: string; error_code: string | null }>("SELECT state,error_code FROM persistence_effects WHERE request_id=$1::uuid AND kind='embedding'", [row.id]);
    if (mismatched) {
      expect(explicitCalls).toBe(0);
      expect(openaiCalls).toBe(0);
      expect(bodies.length).toBe(0);
      expect(effect!.state).not.toBe('committed');
      expect(effect!.error_code).toBe('discovery_schema_mismatch');
      expect((await stored()).every(chunk => chunk.dims === null && chunk.hash === null)).toBe(true);
    } else {
      await assertGoogle();
      expect(effect!.state).toBe('committed');
    }
  });
  async function heldFixture(complete: boolean) {
    const body = 'Synthetic held resource. '.repeat(50_000);
    await engine.putPage('notes/held', { type: 'note', title: 'Held fixture', compiled_truth: body });
    await installFixtureChunks(engine, 'notes/held', [{ chunk_index: 0, chunk_source: 'compiled_truth', chunk_text: 'x'.repeat(2000),
      ...(complete && { embedding: new Float32Array(unit(1)), model: MODEL }) }]);
    await engine.setPageEmbeddingSignature('notes/held', { signature: 'google:gemini-embedding-2:1536' });
    const [page] = await engine.executeRaw<{ id: number }>("SELECT id FROM pages WHERE slug='notes/held'");
    await holdPageEmbedding(engine, page!.id, 'upgrade_backfill');
  }
  test('completed Discovery holds do not produce CLI or doctor approval warnings from primary signatures', async () => {
    await heldFixture(true);
    expect((await readBudgetLimited(engine, { signature: currentEmbeddingSignature() })).pending_chunks).toBe(0);
    const result = {};
    await noteBudgetLimited(engine, [], result, true);
    expect(result).toEqual({});
    expect(await partialEmbeddingEntry.run({ engine, progress: { heartbeat() {} } } as unknown as DoctorContext))
      .toMatchObject([{ name: 'partial_embedding', status: 'ok' }]);
  });
  test('held Discovery approval estimates use the actual Google price', async () => {
    await heldFixture(false);
    const found = await budgetLimitedReport(engine, { signature: currentEmbeddingSignature() });
    expect(found!.report.by_source[0]!.est_usd).toBeCloseTo(572 / 1_000_000 * 0.20, 12);
  });
  test('backfill blockers use the same completed Discovery signature as the drain', async () => {
    await seed(false);
    expect(await readEmbedBackfillBlockers(engine, 'default', currentEmbeddingSignature()))
      .toMatchObject({ remaining: 0, blocked: 0, signature: 'google:gemini-embedding-2:1536' });
  });
  test('sync cost gate counts and prices only the actual Discovery backlog', async () => {
    await seed(false);
    const context = { sources: [], mode: 'deferred' as const, dryRun: false, jsonOut: true, yesFlag: false, full: false,
      label: 'sync', workerSurface: { status: 'no_worker_surface' as const, engineKind: 'pglite' as const }, autoSubmitEnabled: true };
    expect(await runInlineCostGate(engine, context)).toMatchObject({ notice: { staleChars: 0, staleCostUsd: 0, model: 'gemini-embedding-2' } });
    await engine.executeRaw('UPDATE content_chunks SET embedding_discovery=NULL,embedded_at=NULL');
    const [{ chars }] = await engine.executeRaw<{ chars: number }>('SELECT sum(length(chunk_text))::int AS chars FROM content_chunks');
    const result = await runInlineCostGate(engine, context);
    expect(result).toMatchObject({ notice: { staleChars: chars, model: 'gemini-embedding-2' } });
    expect(result.action === 'proceed' ? result.notice?.staleCostUsd : undefined).toBeCloseTo(Math.ceil(chars / 3.5) / 1_000_000 * 0.20, 12);
  });
  test('oversized backfill preview prices the descriptor rather than primary signature', async () => {
    const body = 'Synthetic oversized resource. '.repeat(24_000);
    await engine.putPage('notes/oversized', { type: 'note', title: 'Oversized fixture', compiled_truth: body,
      frontmatter: { embed_skip: { reason: 'oversized' } } });
    const result = await runOversizedBackfill(engine, { dryRun: true });
    expect(result.candidates).toBe(1);
    expect(result.estimated_usd).toBeCloseTo(Math.ceil(body.length / 3.5) / 1_000_000 * 0.20, 12);
  });
  test('late Discovery installs reject a conflicting actual model before writing provenance', async () => {
    await seed();
    const prepared = (await readProjectionSnapshot(engine, 'notes/discovery', 'default'))!;
    await expect(installPageEmbeddings(engine, prepared, prepared.chunks.map(chunk => ({ chunk_index: chunk.chunk_index, chunk_text: chunk.chunk_text, chunk_source: chunk.chunk_source,
      embedding: new Float32Array(unit(0)), model: 'openai:text-embedding-3-large' }))))
      .rejects.toMatchObject({ code: 'discovery_schema_mismatch' });
    expect((await stored()).every(chunk => chunk.dims === null && chunk.hash === null)).toBe(true);
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
