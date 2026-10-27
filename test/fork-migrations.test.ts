// Protects fork/upstream ledger separation, ordering, replay and future adoption.
// A fork stamp in config.version would skip an unrelated future upstream change.
import { afterAll, beforeAll, describe, expect, test } from 'bun:test';
import { PGLiteEngine } from '../src/core/pglite-engine.ts';
import type { BrainEngine } from '../src/core/engine.ts';
import { MIGRATIONS, runMigrations } from '../src/core/migrate.ts';
import { LATEST_SCHEMA_VERSION } from '../src/core/schema-migrations/latest.generated.ts';
import { hasPendingMigrations } from '../src/core/migrate-pending.ts';
import { isolatedPersistencePostgres } from './helpers/persistence-postgres.ts';
import { testBackends } from './helpers/test-backends.ts';
import { withEnv } from './helpers/with-env.ts';

const key = 'fork_migration.sinaxis-001-google-discovery-embedding';
for (const backend of testBackends()) describe(`fork migration ledger (${backend})`, () => {
  let engine: BrainEngine;
  let close: () => Promise<void>;
  beforeAll(async () => {
    if (backend === 'postgres') ({ engine, close } = await isolatedPersistencePostgres(process.env.DATABASE_URL!));
    else { const lite = new PGLiteEngine(); await lite.connect({}); await lite.initSchema(); engine = lite; close = () => lite.disconnect(); }
  }, 120_000);
  afterAll(async () => { await close?.(); });
  async function resetFork() {
    await engine.executeRaw('DROP INDEX IF EXISTS idx_chunks_embedding_discovery');
    await engine.executeRaw('ALTER TABLE content_chunks DROP COLUMN IF EXISTS embedding_discovery');
    await engine.executeRaw('DELETE FROM config WHERE key IN ($1, $2)', [key, 'embedding_columns']);
    await engine.setConfig('version', String(LATEST_SCHEMA_VERSION));
  }
  async function indexOid() {
    return (await engine.executeRaw<{ oid: number }>("SELECT to_regclass('idx_chunks_embedding_discovery')::oid AS oid"))[0]!.oid;
  }
  test('a fresh brain records the fork independently of the current upstream version', async () => {
    expect(await engine.getConfig('version')).toBe(String(LATEST_SCHEMA_VERSION));
    expect(await engine.getConfig(key)).toBe('applied');
    expect(await hasPendingMigrations(engine)).toBe(false);
  });
  test('opening an upstream brain at the current version bootstraps discovery and applies only the fork ledger', async () => {
    const previous = engine;
    const cold = backend === 'pglite' ? new PGLiteEngine() : null;
    try {
      if (cold) {
        // A historical schema must execute initSchema instead of the baked snapshot fast path.
        await withEnv({ GBRAIN_PGLITE_SNAPSHOT: undefined }, async () => { await cold.connect({}); await cold.initSchema(); });
        engine = cold;
      }
      await resetFork();
      expect(await hasPendingMigrations(engine)).toBe(true);
      await engine.initSchema();
      expect(await engine.getConfig('version')).toBe(String(LATEST_SCHEMA_VERSION));
      expect(await engine.getConfig(key)).toBe('applied');
      expect(await indexOid()).toBeGreaterThan(0);
      expect(await engine.executeRaw(`SELECT format_type(atttypid, atttypmod) AS type
        FROM pg_attribute WHERE attrelid='content_chunks'::regclass AND attname='embedding_discovery' AND NOT attisdropped`)).toEqual([{ type: 'vector(1536)' }]);
    } finally { engine = previous; await cold?.disconnect(); }
  });
  test('a different future upstream migration runs before the fork and never shares its stamp', async () => {
    await resetFork();
    const future = { version: LATEST_SCHEMA_VERSION + 1, name: 'future_unrelated', sql: '', handler: async (e: BrainEngine) => {
      expect(await e.getConfig(key)).toBeNull();
      expect((await e.executeRaw<{ present: boolean }>("SELECT EXISTS(SELECT 1 FROM pg_attribute WHERE attrelid='content_chunks'::regclass AND attname='embedding_discovery' AND NOT attisdropped) AS present"))[0]!.present).toBe(false);
      await e.setConfig('future_upstream_probe', 'ran');
    } };
    MIGRATIONS.push(future);
    try {
      await runMigrations(engine);
      expect(await engine.getConfig('version')).toBe(String(LATEST_SCHEMA_VERSION + 1));
      expect(await engine.getConfig('future_upstream_probe')).toBe('ran');
      expect(await engine.getConfig(key)).toBe('applied');
      await runMigrations(engine);
      expect(await engine.getConfig('version')).toBe(String(LATEST_SCHEMA_VERSION + 1));
    } finally { MIGRATIONS.splice(MIGRATIONS.indexOf(future), 1); await engine.executeRaw("DELETE FROM config WHERE key='future_upstream_probe'"); }
  });
  test('later upstream adoption of the same DDL is a no-op over the fork schema', async () => {
    await engine.setConfig('version', String(LATEST_SCHEMA_VERSION));
    const oid = await indexOid();
    const adoption = { version: LATEST_SCHEMA_VERSION + 1, name: 'future_adoption', sql: `
      ALTER TABLE content_chunks ADD COLUMN IF NOT EXISTS embedding_discovery vector(1536);
      CREATE INDEX IF NOT EXISTS idx_chunks_embedding_discovery ON content_chunks USING hnsw (embedding_discovery vector_cosine_ops) WHERE embedding_discovery IS NOT NULL;
    ` };
    MIGRATIONS.push(adoption);
    try {
      await runMigrations(engine);
      expect(await engine.getConfig('version')).toBe(String(LATEST_SCHEMA_VERSION + 1));
      expect(await engine.getConfig(key)).toBe('applied');
      expect(await indexOid()).toBe(oid);
    } finally { MIGRATIONS.splice(MIGRATIONS.indexOf(adoption), 1); }
  });
  test('a lost fork receipt safely replays DDL while preserving custom registry and index identity', async () => {
    const registry = { embedding_discovery: { type: 'vector', dimensions: 1536, provider: 'google:gemini-embedding-2' }, embedding_custom: { dimensions: 129, provider: 'example' } };
    await engine.setConfig('embedding_columns', JSON.stringify(registry));
    const oid = await indexOid();
    for (let i = 0; i < 2; i++) {
      await engine.executeRaw('DELETE FROM config WHERE key=$1', [key]);
      await runMigrations(engine);
      expect(await engine.getConfig(key)).toBe('applied');
      expect(await engine.getConfig('version')).toBe(String(LATEST_SCHEMA_VERSION + 1));
      expect(await indexOid()).toBe(oid);
      expect(JSON.parse((await engine.getConfig('embedding_columns'))!)).toEqual(registry);
    }
    await engine.setConfig('version', String(LATEST_SCHEMA_VERSION));
  });
  test('an existing discovery vector with a different width is refused without a receipt', async () => {
    await resetFork();
    await engine.executeRaw('ALTER TABLE content_chunks ADD COLUMN embedding_discovery vector(768)');
    try {
      await expect(runMigrations(engine)).rejects.toThrow('Fork discovery schema verification failed');
      expect(await engine.getConfig(key)).toBeNull();
      expect(await engine.getConfig('version')).toBe(String(LATEST_SCHEMA_VERSION));
    } finally { await resetFork(); await runMigrations(engine); }
  });
  // Old experimental receipts used the same identifier. Existing width rejection
  // only covered missing receipts; no production-only test seam is needed here.
  test('an applied 768d receipt is pending and fails verification without changing legacy data', async () => {
    await resetFork();
    await engine.executeRaw('ALTER TABLE content_chunks ADD COLUMN embedding_discovery vector(768)');
    await engine.executeRaw('CREATE INDEX idx_chunks_embedding_discovery ON content_chunks USING hnsw (embedding_discovery vector_cosine_ops) WHERE embedding_discovery IS NOT NULL');
    const column = { name: 'embedding_discovery', type: 'vector' as const, dimensions: 768, embeddingModel: 'google:gemini-embedding-2' };
    const registry = JSON.stringify({ embedding_discovery: { type: column.type, dimensions: column.dimensions, provider: column.embeddingModel } });
    await engine.setConfig('embedding_columns', registry);
    await engine.setConfig(key, 'applied');
    await engine.putPage('resource/legacy', { type: 'concept', title: 'Legacy resource', compiled_truth: 'Legacy resource content.' });
    await engine.upsertChunks('resource/legacy', [{ chunk_index: 0, chunk_text: 'Legacy resource content.', chunk_source: 'compiled_truth' }],
      { embeddingColumn: { name: 'embedding', type: 'vector', dimensions: 1536, embeddingModel: 'openai:text-embedding-3-large' } });
    await engine.executeRaw(`UPDATE content_chunks SET embedding_discovery=$1::text::vector WHERE page_id=(SELECT id FROM pages WHERE slug='resource/legacy')`,
      [JSON.stringify(Array(768).fill(1))]);
    const oid = await indexOid();
    const readVector = () => engine.executeRaw(`SELECT embedding_discovery::text AS vector FROM content_chunks cc JOIN pages p ON p.id=cc.page_id WHERE p.slug='resource/legacy'`);
    const before = await readVector();
    expect(before).toHaveLength(1);
    try {
      expect(await hasPendingMigrations(engine)).toBe(true);
      await expect(runMigrations(engine)).rejects.toThrow('Fork discovery schema verification failed');
      expect(await engine.getConfig(key)).toBe('applied');
      expect(await engine.getConfig('version')).toBe(String(LATEST_SCHEMA_VERSION));
      expect(await engine.getConfig('embedding_columns')).toBe(registry);
      expect(await indexOid()).toBe(oid);
      expect(await readVector()).toEqual(before);
    } finally { await resetFork(); await runMigrations(engine); }
  });
  test('a failed fork build leaves no receipt and succeeds after repair', async () => {
    await resetFork();
    await engine.executeRaw('ALTER TABLE content_chunks ADD COLUMN embedding_discovery text');
    await expect(runMigrations(engine)).rejects.toThrow();
    expect(await engine.getConfig(key)).toBeNull();
    expect(await engine.getConfig('version')).toBe(String(LATEST_SCHEMA_VERSION));
    await engine.executeRaw('ALTER TABLE content_chunks DROP COLUMN embedding_discovery');
    await runMigrations(engine);
    expect(await engine.getConfig(key)).toBe('applied');
  });

});
