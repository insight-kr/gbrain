import type { Migration } from './types.ts';
import { buildIndexOnline } from './helpers.ts';

/** A separate space preserves every existing text and Voyage vector. */
export const v276: Migration = {
  version: 276,
  name: 'google_discovery_embedding',
  idempotent: true,
  transaction: false,
  sql: `
    ALTER TABLE content_chunks ADD COLUMN IF NOT EXISTS embedding_discovery vector(768);
    INSERT INTO config (key, value) VALUES ('embedding_columns',
      '{"embedding_discovery":{"type":"vector","dimensions":768,"provider":"google:gemini-embedding-2"}}')
    ON CONFLICT (key) DO UPDATE SET value = (EXCLUDED.value::jsonb || CASE WHEN jsonb_typeof(gbrain_try_jsonb(config.value)) = 'object' THEN gbrain_try_jsonb(config.value) ELSE '{}'::jsonb END)::text;
  `,
  handler: async engine => {
    await buildIndexOnline(engine, 276, { name: 'idx_chunks_embedding_discovery', table: 'content_chunks',
      sql: 'CREATE INDEX IF NOT EXISTS idx_chunks_embedding_discovery ON content_chunks USING hnsw (embedding_discovery vector_cosine_ops) WHERE embedding_discovery IS NOT NULL' });
  },
  verify: async engine => {
    const [row] = await engine.executeRaw<{ valid: boolean }>(`SELECT
      EXISTS(SELECT 1 FROM pg_attribute WHERE attrelid='content_chunks'::regclass AND attname='embedding_discovery' AND format_type(atttypid,atttypmod)='vector(768)')
      AND EXISTS(SELECT 1 FROM pg_index WHERE indexrelid=to_regclass('idx_chunks_embedding_discovery') AND indisvalid)
      AND EXISTS(SELECT 1 FROM config WHERE key='embedding_columns' AND value::jsonb->'embedding_discovery' =
        '{"type":"vector","dimensions":768,"provider":"google:gemini-embedding-2"}'::jsonb) AS valid`);
    return row?.valid === true;
  },
};
