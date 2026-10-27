import type { BrainEngine } from './engine.ts';
import { buildIndexOnline } from './schema-migrations/helpers.ts';
import { FORK_MIGRATION_SEEDS, assertDiscoveryColumn } from './fork-migration-state.ts';

/** Additive DDL can later be adopted by an upstream numbered migration unchanged. */
export const DISCOVERY_FORK_SQL = `
    ALTER TABLE content_chunks ADD COLUMN IF NOT EXISTS embedding_discovery vector(1536);
    INSERT INTO config (key, value) VALUES ('embedding_columns',
      '{"embedding_discovery":{"type":"vector","dimensions":1536,"provider":"google:gemini-embedding-2"}}')
    ON CONFLICT (key) DO UPDATE SET value = (EXCLUDED.value::jsonb || CASE WHEN jsonb_typeof(gbrain_try_jsonb(config.value)) = 'object' THEN gbrain_try_jsonb(config.value) ELSE '{}'::jsonb END)::text;
  `;

export async function applyDiscoveryFork(engine: BrainEngine): Promise<void> {
  // runMigration's numeric argument is an unused statement tag on both engines;
  // zero is not a schema version and never writes the upstream version counter.
  await engine.transaction(tx => tx.runMigration(0, DISCOVERY_FORK_SQL));
  await buildIndexOnline(engine, 0, { name: 'idx_chunks_embedding_discovery', table: 'content_chunks',
    sql: 'CREATE INDEX IF NOT EXISTS idx_chunks_embedding_discovery ON content_chunks USING hnsw (embedding_discovery vector_cosine_ops) WHERE embedding_discovery IS NOT NULL' });
  await assertDiscoveryColumn(engine, 'embedding_discovery');
}

/** Called only after the upstream runner has completed successfully. */
export async function runForkMigrations(engine: BrainEngine): Promise<void> {
  for (const [key, value] of Object.entries(FORK_MIGRATION_SEEDS)) {
    if (await engine.getConfig(key) === value) {
      await assertDiscoveryColumn(engine, 'embedding_discovery');
      continue;
    }
    await applyDiscoveryFork(engine);
    // A failed/interrupted DDL or index build leaves no receipt. Replay is safe.
    await engine.setConfig(key, value);
  }
}
