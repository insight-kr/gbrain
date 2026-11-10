/**
 * Protects durable publication after real process death, not a thrown-error simulation.
 * A committed partial publication or an unretryable journal fails this case; existing synthetic tests
 * do not kill a Discovery producer. Uses the existing native fault hook, no new production seam.
 */
import { expect, test } from 'bun:test';
import { fileURLToPath } from 'node:url';
import { isolatedPersistencePostgres } from '../helpers/persistence-postgres.ts';
import { pathologicalArms } from './harness.ts';
for (const { backend, databaseUrl } of pathologicalArms()) {
  if (backend !== 'postgres') continue;
  test('Discovery SIGKILL rolls back and a fresh process publishes the complete receipt (postgres)', async () => {
    const { engine, databaseUrl: url, close } = await isolatedPersistencePostgres(databaseUrl!);
    try {
      const run = (mode: string) => Bun.spawn([process.execPath, fileURLToPath(new URL('../helpers/sinaxis-discovery-crash-worker.ts', import.meta.url)), mode],
        { env: { ...process.env, DATABASE_URL: url }, stdout: 'pipe', stderr: 'pipe' });
      const crashed = run('crash');
      await new Response(crashed.stderr).text();
      expect(await crashed.exited).not.toBe(0);
      expect(await engine.executeRaw('SELECT id FROM pages')).toHaveLength(0);
      expect(await engine.executeRaw("SELECT id FROM derived_generations WHERE state='live'")).toHaveLength(0);
      const restarted = run('retry');
      const output = await new Response(restarted.stdout).text();
      const errors = await new Response(restarted.stderr).text();
      expect({ rc: await restarted.exited, errors }).toMatchObject({ rc: 0 });
      expect(JSON.parse(output.trim())).toMatchObject({ generation: 1, complete: true, revision: 'crash-revision' });
    } finally { await close(); }
  }, 120_000);
}
