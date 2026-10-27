// Protects independent primary/Discovery widths in generated SQL. A broad
// 1536 substitution retargets Discovery and breaks non-1536 Postgres init;
// existing checks only looked for the primary width. No new seam is needed.
import { describe, test, expect } from 'bun:test';
import { getPGLiteSchema, PGLITE_SCHEMA_SQL } from '../../src/core/pglite-schema.ts';
import { getPostgresSchema } from '../../src/core/postgres-engine.ts';

describe('getPGLiteSchema', () => {
  test('default produces gateway-default schema (v0.37+: 1024d + voyage:voyage-4)', () => {
    // v0.37 fix wave Lane A.1 + CDX2-1: defaults now track the canonical
    // gateway constants in `ai/defaults.ts` instead of the stale v0.13
    // OpenAI literals (1536 / text-embedding-3-large). Fixes the
    // headline bug where bare `gbrain init --pglite` produced a 1536
    // schema while the Voyage default model emitted 1024-dim vectors.
    const sql = getPGLiteSchema();
    expect(sql).toMatch(/vector\(1024\)/);
    expect(sql).toMatch(/'voyage:voyage-4'/);
    expect(sql).not.toMatch(/__EMBEDDING_DIMS__/);
    expect(sql).not.toMatch(/__EMBEDDING_MODEL__/);
  });

  test('Gemini 768d substitution', () => {
    const sql = getPGLiteSchema(768, 'gemini-embedding-001');
    expect(sql).toMatch(/vector\(768\)/);
    expect(sql).toMatch(/'gemini-embedding-001'/);
    expect(sql).toMatch(/\('embedding_model', 'gemini-embedding-001'\)/);
    expect(sql).toMatch(/\('embedding_dimensions', '768'\)/);
    expect(sql).toMatch(/^\s+embedding\s+vector\(768\)/m);
    expect(sql).toMatch(/^\s+embedding_discovery\s+vector\(1536\)/m);
  });

  test('Voyage 1024d substitution', () => {
    const sql = getPGLiteSchema(1024, 'voyage-3-large');
    expect(sql).toMatch(/vector\(1024\)/);
    expect(sql).toMatch(/'voyage-3-large'/);
    expect(sql).toMatch(/\('embedding_model', 'voyage-3-large'\)/);
    expect(sql).toMatch(/\('embedding_dimensions', '1024'\)/);
    expect(sql).toContain('idx_chunks_embedding ON content_chunks USING hnsw');
  });

  test('Voyage 2048d skips unsupported HNSW index but keeps vector column', () => {
    const sql = getPGLiteSchema(2048, 'voyage-4-large');
    expect(sql).toMatch(/vector\(2048\)/);
    expect(sql).toMatch(/'voyage-4-large'/);
    expect(sql).toMatch(/\('embedding_dimensions', '2048'\)/);
    expect(sql).not.toContain('idx_chunks_embedding ON content_chunks USING hnsw');
    expect(sql).toContain('exact vector scans remain available');
  });

  test('PGLITE_SCHEMA_SQL back-compat constant is the default-dim schema', () => {
    expect(PGLITE_SCHEMA_SQL).toBe(getPGLiteSchema());
  });
});

for (const [name, render] of [['PGLite', getPGLiteSchema], ['Postgres', getPostgresSchema]] as const) {
  for (const dims of [768, 1024, 2048, 3072]) test(`${name} primary ${dims}d keeps Discovery at 1536d`, () => {
    const sql = render(dims, 'synthetic-model');
    expect(sql.match(/^\s+embedding\s+[^\n]+/m)?.[0]).toContain(`vector(${dims})`);
    expect(sql.match(/^\s+embedding_discovery\s+[^\n]+/m)?.[0]).toContain('vector(1536)');
    expect(sql).toContain('ON content_chunks USING hnsw (embedding_discovery vector_cosine_ops)');
  });
}

describe('getPostgresSchema', () => {
  test('Voyage 2048d updates vector column and seeded config but skips HNSW', () => {
    const sql = getPostgresSchema(2048, 'voyage-4-large');
    expect(sql).toMatch(/vector\(2048\)/);
    expect(sql).toMatch(/\('embedding_model', 'voyage-4-large'\)/);
    expect(sql).toMatch(/\('embedding_dimensions', '2048'\)/);
    expect(sql).not.toContain('idx_chunks_embedding ON content_chunks USING hnsw');
  });

  test('escapes configured model before inserting into schema SQL literals', () => {
    const sql = getPostgresSchema(1024, "voyage-weird'quoted");
    expect(sql).toContain("'voyage-weird''quoted'");
    expect(sql).not.toContain("'voyage-weird'quoted'");
  });
});
