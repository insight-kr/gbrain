// Protects upgrade completeness and forgiving registry merge, using real migration SQL.
// Casting malformed config can abort the fork migration or discard custom column registrations.
// Fresh-schema coverage does not start with malformed registries or replay the fork migration.
import { afterAll, beforeAll, describe, expect, test } from 'bun:test';
import { PGLiteEngine } from '../src/core/pglite-engine.ts';
import type { BrainEngine } from '../src/core/engine.ts';
import { isolatedPersistencePostgres } from './helpers/persistence-postgres.ts';
import { testBackends } from './helpers/test-backends.ts';
import { applyDiscoveryFork } from '../src/core/fork-migrations.ts';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { saveConfig } from '../src/core/config.ts';
import { withEnv } from './helpers/with-env.ts';
import { installFixtureChunks } from './helpers/page-projection.ts';

for (const backend of testBackends()) describe(`Google discovery migration (${backend})`, () => {
let engine: BrainEngine;
let close: () => Promise<void>;
beforeAll(async () => {
  if (backend === 'postgres') ({ engine, close } = await isolatedPersistencePostgres(process.env.DATABASE_URL!));
  else { const lite = new PGLiteEngine(); await lite.connect({}); await lite.initSchema(); engine = lite; close = () => lite.disconnect(); }
}, 120_000);
afterAll(async () => { await close?.(); });
const descriptor = { type: 'vector', dimensions: 1536, provider: 'google:gemini-embedding-2' };

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

// The ordinary CLI warns on migration failure and continues. Its Discovery
// notice must identify this refusal without blocking unrelated Wiki searches.
if (backend === 'pglite') test('CLI startup names the Discovery refusal and still serves Wiki text', async () => {
  const home = mkdtempSync(join(tmpdir(), 'discovery-cli-'));
  const path = join(home, '.gbrain', 'brain.pglite');
  const lite = new PGLiteEngine();
  try {
    await withEnv({ GBRAIN_HOME: home }, async () => {
      saveConfig({ engine: 'pglite', database_path: path, embedding_model: 'openai:text-embedding-3-large', embedding_dimensions: 1536 });
      await lite.connect({ engine: 'pglite', database_path: path });
      await lite.initSchema();
      await lite.putPage('wiki/synthetic-marker', { type: 'concept', title: 'Synthetic marker', compiled_truth: 'Synthetic marker text.' });
      await installFixtureChunks(lite, 'wiki/synthetic-marker', [{ chunk_index: 0, chunk_text: 'Synthetic marker text.', chunk_source: 'compiled_truth' }]);
      await lite.executeRaw('ALTER TABLE content_chunks ALTER COLUMN embedding_discovery TYPE vector(768) USING NULL');
      await lite.disconnect();
    });
    const child = Bun.spawnSync([process.execPath, '--no-env-file', join(import.meta.dir, '../src/cli.ts'), 'search', 'synthetic marker', '--json'],
      { cwd: home, env: { ...process.env, GBRAIN_HOME: home, GBRAIN_SKIP_STARTUP_HOOKS: '1', GBRAIN_NO_RETRY_CONNECT: '1' }, stdout: 'pipe', stderr: 'pipe', timeout: 60_000 });
    expect(child.exitCode, child.stderr.toString()).toBe(0);
    expect(child.stdout.toString()).toContain('wiki/synthetic-marker');
    expect(child.stderr.toString()).toContain('discovery_schema_mismatch');
    expect(child.stderr.toString()).not.toContain('migrations_pending');
  } finally { await lite.disconnect(); rmSync(home, { recursive: true, force: true }); }
}, 120_000);

});
