import type { BrainEngine } from './engine.ts';
import type { ResolvedColumn } from './types.ts';
import { opError } from './ops/contract.ts';

/** Per-brain fork receipts are separate from upstream config.version. */
export const FORK_MIGRATION_SEEDS: Readonly<Record<string, string>> = {
  'fork_migration.sinaxis-001-google-discovery-embedding': 'applied',
};

export function isForkMigrationSeed(key: string, value: string): boolean {
  return Object.hasOwn(FORK_MIGRATION_SEEDS, key) && FORK_MIGRATION_SEEDS[key] === value;
}

/** A receipt alone cannot certify an old experimental discovery schema. */
export async function hasValidDiscoveryFork(engine: Pick<BrainEngine, 'executeRaw'>): Promise<boolean> {
  const [row] = await engine.executeRaw<{ valid: boolean }>(`SELECT
    EXISTS(SELECT 1 FROM pg_attribute WHERE attrelid='content_chunks'::regclass AND attname='embedding_discovery' AND NOT attisdropped AND format_type(atttypid,atttypmod)='vector(1536)')
    AND EXISTS(SELECT 1 FROM pg_index i
      JOIN pg_class c ON c.oid=i.indexrelid JOIN pg_am am ON am.oid=c.relam
      JOIN pg_attribute a ON a.attrelid=i.indrelid AND a.attnum=i.indkey[0]
      JOIN pg_opclass op ON op.oid=i.indclass[0]
      WHERE i.indexrelid=to_regclass('idx_chunks_embedding_discovery') AND i.indisvalid
        AND i.indrelid='content_chunks'::regclass AND i.indnkeyatts=1 AND a.attname='embedding_discovery'
        AND am.amname='hnsw' AND op.opcname='vector_cosine_ops')
    AND EXISTS(SELECT 1 FROM config WHERE key='embedding_columns' AND
      (gbrain_try_jsonb(value)->'embedding_discovery') @> '{"type":"vector","dimensions":1536,"provider":"google:gemini-embedding-2"}'::jsonb) AS valid`);
  return row?.valid === true;
}

/** Guard Discovery at use time; unrelated Wiki vectors need no fork probe. */
export async function assertDiscoveryColumn(engine: Pick<BrainEngine, 'executeRaw'>, column: string | ResolvedColumn | undefined): Promise<void> {
  if ((typeof column === 'string' ? column : column?.name) !== 'embedding_discovery') return;
  const descriptorValid = typeof column === 'string' || column?.type === 'vector' && column.dimensions === 1536 && column.embeddingModel === 'google:gemini-embedding-2';
  if (!descriptorValid || !await hasValidDiscoveryFork(engine)) throw opError('discovery_schema_mismatch',
    'Fork discovery schema verification failed: expected vector(1536), its Google registry descriptor and a valid discovery HNSW cosine index; receipt was not changed. Automatic conversion is not supported.',
    'Use a brain initialized with the 1536-dimensional Discovery schema. An experimental 768-dimensional brain needs a separately approved migration. Wiki text remains available.');
}
