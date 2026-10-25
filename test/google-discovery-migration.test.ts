// Protects upgrade completeness and forgiving registry merge, using real migration SQL.
// Reusing upstream v267 or casting malformed config loses indexes or aborts upgrades.
// Existing fresh-schema coverage never starts with broken registry data or the old fork stamp.
import { afterAll, beforeAll, describe, expect, test } from 'bun:test';
import { PGLiteEngine } from '../src/core/pglite-engine.ts';
import type { BrainEngine } from '../src/core/engine.ts';
import { isolatedPersistencePostgres } from './helpers/persistence-postgres.ts';
import { testBackends } from './helpers/test-backends.ts';
import { MIGRATIONS } from '../src/core/migrate.ts';

for (const backend of testBackends()) describe(`Google discovery migration (${backend})`, () => {
let engine: BrainEngine;
let close: () => Promise<void>;
beforeAll(async () => {
  if (backend === 'postgres') ({ engine, close } = await isolatedPersistencePostgres(process.env.DATABASE_URL!));
  else { const lite = new PGLiteEngine(); await lite.connect({}); await lite.initSchema(); engine = lite; close = () => lite.disconnect(); }
}, 120_000);
afterAll(async () => { await close?.(); });
const discovery = MIGRATIONS.find(m => m.name === 'google_discovery_embedding')!;
const descriptor = { type: 'vector', dimensions: 768, provider: 'google:gemini-embedding-2' };

for (const value of ['invalid json', '[]', 'null', '42', '"scalar"']) test(`discovery migration repairs ignored non-object registry: ${value}`, async () => {
  await engine.setConfig('embedding_columns', value);
  await engine.runMigration(discovery.version, discovery.sql);
  await discovery.handler?.(engine);
  expect(JSON.parse((await engine.getConfig('embedding_columns'))!)).toEqual({ embedding_discovery: descriptor });
  expect(await discovery.verify?.(engine)).toBe(true);
});

test('discovery migration preserves custom registry entries on replay', async () => {
  const custom = { embedding_custom: { type: 'vector', dimensions: 1024, provider: 'voyage:voyage-3-large' } };
  await engine.setConfig('embedding_columns', JSON.stringify(custom));
  for (let i = 0; i < 2; i++) { await engine.runMigration(discovery.version, discovery.sql); await discovery.handler?.(engine); }
  expect(JSON.parse((await engine.getConfig('embedding_columns'))!)).toEqual({ ...custom, embedding_discovery: descriptor });
});

});
