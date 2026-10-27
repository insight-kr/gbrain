// Protects fork/upstream ledger separation, ordering, replay and future adoption.
// A fork stamp in config.version would skip an unrelated future upstream change.
import { afterAll, beforeAll, describe, expect, test } from 'bun:test';
import { PGLiteEngine } from '../src/core/pglite-engine.ts';
import type { BrainEngine } from '../src/core/engine.ts';
import { MIGRATIONS, runMigrations } from '../src/core/migrate.ts';
import { hasPendingMigrations } from '../src/core/migrate-pending.ts';
import { isolatedPersistencePostgres } from './helpers/persistence-postgres.ts';
import { testBackends } from './helpers/test-backends.ts';

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
    await engine.setConfig('version', '275');
  }
  async function indexOid() {
    return (await engine.executeRaw<{ oid: number }>("SELECT to_regclass('idx_chunks_embedding_discovery')::oid AS oid"))[0]!.oid;
  }
  test('a fresh brain records the fork independently of upstream v275', async () => {
    expect(await engine.getConfig('version')).toBe('275');
    expect(await engine.getConfig(key)).toBe('applied');
    expect(await hasPendingMigrations(engine)).toBe(false);
  });
  test('opening an upstream v275 brain bootstraps discovery and applies only the fork ledger', async () => {
    await resetFork();
    expect(await hasPendingMigrations(engine)).toBe(true);
    await engine.initSchema();
    expect(await engine.getConfig('version')).toBe('275');
    expect(await engine.getConfig(key)).toBe('applied');
    expect(await indexOid()).toBeGreaterThan(0);
  });
  test('a different future upstream v276 runs before the fork and never shares its stamp', async () => {
    await resetFork();
    const future = { version: 276, name: 'future_unrelated', sql: '', handler: async (e: BrainEngine) => {
      expect(await e.getConfig(key)).toBeNull();
      expect((await e.executeRaw<{ present: boolean }>("SELECT EXISTS(SELECT 1 FROM pg_attribute WHERE attrelid='content_chunks'::regclass AND attname='embedding_discovery' AND NOT attisdropped) AS present"))[0]!.present).toBe(false);
      await e.setConfig('future_upstream_probe', 'ran');
    } };
    MIGRATIONS.push(future);
    try {
      await runMigrations(engine);
      expect(await engine.getConfig('version')).toBe('276');
      expect(await engine.getConfig('future_upstream_probe')).toBe('ran');
      expect(await engine.getConfig(key)).toBe('applied');
      await runMigrations(engine);
      expect(await engine.getConfig('version')).toBe('276');
    } finally { MIGRATIONS.splice(MIGRATIONS.indexOf(future), 1); await engine.executeRaw("DELETE FROM config WHERE key='future_upstream_probe'"); }
  });
  test('later upstream adoption of the same DDL is a no-op over the fork schema', async () => {
    await engine.setConfig('version', '275');
    const oid = await indexOid();
    const adoption = { version: 276, name: 'future_adoption', sql: `
      ALTER TABLE content_chunks ADD COLUMN IF NOT EXISTS embedding_discovery vector(768);
      CREATE INDEX IF NOT EXISTS idx_chunks_embedding_discovery ON content_chunks USING hnsw (embedding_discovery vector_cosine_ops) WHERE embedding_discovery IS NOT NULL;
    ` };
    MIGRATIONS.push(adoption);
    try {
      await runMigrations(engine);
      expect(await engine.getConfig('version')).toBe('276');
      expect(await engine.getConfig(key)).toBe('applied');
      expect(await indexOid()).toBe(oid);
    } finally { MIGRATIONS.splice(MIGRATIONS.indexOf(adoption), 1); }
  });
  test('a lost fork receipt safely replays DDL while preserving custom registry and index identity', async () => {
    const registry = { embedding_discovery: { type: 'vector', dimensions: 768, provider: 'google:gemini-embedding-2' }, embedding_custom: { dimensions: 129, provider: 'example' } };
    await engine.setConfig('embedding_columns', JSON.stringify(registry));
    const oid = await indexOid();
    for (let i = 0; i < 2; i++) {
      await engine.executeRaw('DELETE FROM config WHERE key=$1', [key]);
      await runMigrations(engine);
      expect(await engine.getConfig(key)).toBe('applied');
      expect(await engine.getConfig('version')).toBe('276');
      expect(await indexOid()).toBe(oid);
      expect(JSON.parse((await engine.getConfig('embedding_columns'))!)).toEqual(registry);
    }
    await engine.setConfig('version', '275');
  });
  test('a failed fork build leaves no receipt and succeeds after repair', async () => {
    await resetFork();
    await engine.executeRaw('ALTER TABLE content_chunks ADD COLUMN embedding_discovery text');
    await expect(runMigrations(engine)).rejects.toThrow();
    expect(await engine.getConfig(key)).toBeNull();
    expect(await engine.getConfig('version')).toBe('275');
    await engine.executeRaw('ALTER TABLE content_chunks DROP COLUMN embedding_discovery');
    await runMigrations(engine);
    expect(await engine.getConfig(key)).toBe('applied');
  });

});
