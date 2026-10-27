// Protects upgrade completeness and forgiving registry merge, using real migration SQL.
// Casting malformed config can abort the fork migration or discard custom column registrations.
// Fresh-schema coverage does not start with malformed registries or replay the fork migration.
import { afterAll, beforeAll, describe, expect, test } from 'bun:test';
import { PGLiteEngine } from '../src/core/pglite-engine.ts';
import type { BrainEngine } from '../src/core/engine.ts';
import { isolatedPersistencePostgres } from './helpers/persistence-postgres.ts';
import { testBackends } from './helpers/test-backends.ts';
import { applyDiscoveryFork } from '../src/core/fork-migrations.ts';

for (const backend of testBackends()) describe(`Google discovery migration (${backend})`, () => {
let engine: BrainEngine;
let close: () => Promise<void>;
beforeAll(async () => {
  if (backend === 'postgres') ({ engine, close } = await isolatedPersistencePostgres(process.env.DATABASE_URL!));
  else { const lite = new PGLiteEngine(); await lite.connect({}); await lite.initSchema(); engine = lite; close = () => lite.disconnect(); }
}, 120_000);
afterAll(async () => { await close?.(); });
const descriptor = { type: 'vector', dimensions: 768, provider: 'google:gemini-embedding-2' };

for (const value of ['invalid json', '[]', 'null', '42', '"scalar"']) test(`discovery migration repairs ignored non-object registry: ${value}`, async () => {
  await engine.setConfig('embedding_columns', value);
  await applyDiscoveryFork(engine);
  expect(JSON.parse((await engine.getConfig('embedding_columns'))!)).toEqual({ embedding_discovery: descriptor });
});

test('discovery migration preserves custom registry entries on replay', async () => {
  const custom = { embedding_custom: { type: 'vector', dimensions: 1024, provider: 'voyage:voyage-3-large' } };
  await engine.setConfig('embedding_columns', JSON.stringify(custom));
  for (let i = 0; i < 2; i++) { await applyDiscoveryFork(engine); }
  expect(JSON.parse((await engine.getConfig('embedding_columns'))!)).toEqual({ ...custom, embedding_discovery: descriptor });
});

});
