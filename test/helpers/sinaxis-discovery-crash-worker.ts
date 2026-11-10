/** Synthetic child used by the real PostgreSQL crash/restart regression; never reads a host brain. */
import { PostgresEngine } from '../../src/core/postgres-engine.ts';
import { createDiscoveryEntry, discoveryKey } from '../../src/core/sinaxis-discovery/entry.ts';
import { installFaultHook } from '../../src/core/persistence/fault-points.ts';
import { assertSafeE2eDatabaseUrl } from './db-guard.ts';
const databaseUrl = process.env.DATABASE_URL!;
assertSafeE2eDatabaseUrl(databaseUrl);
globalThis.fetch = Object.assign(() => { throw new Error('External fetch forbidden'); }, { preconnect() {} });
const engine = new PostgresEngine();
await engine.connect({ database_url: databaseUrl });
const binding = { revision: 'crash-revision', profileDigest: 'c'.repeat(64), epoch: 1, lease: 'crash-lease' };
const entry = createDiscoveryEntry(engine, {
  withFence: async (_key, run) => run(),
  read: async () => ({ ...binding, enabled: true }),
});
if (process.argv[2] === 'crash') installFaultHook(point => {
  if (point === 'generation:published') process.kill(process.pid, 'SIGKILL');
});
try {
  const key = discoveryKey('default', 'crash-resource');
  await entry.publish({ key, binding, expectedGeneration: 0, items: [{ segmentId: 'one', text: 'Synthetic crash content.' }] });
  console.log(JSON.stringify(await entry.read(key, binding)));
} finally { await engine.disconnect(); }
