/**
 * Protects Discovery and the seven existing stamps on purge preview and delete at the default/minimum hold.
 * Removing the Discovery tuple entry loses its restorable page after 72 hours. Existing tests derive their cases
 * from that tuple, so cannot detect the missing entry. No production-only test seam is needed.
 */
import { afterAll, beforeAll, describe, expect, spyOn, test } from 'bun:test';
import { isolatedSharedSkillsEngine } from '../helpers/shared-skills-engine.ts';
import { pathologicalArms } from './harness.ts';

let calls = 0;
let fetchSpy: ReturnType<typeof spyOn>;
beforeAll(() => { fetchSpy = spyOn(globalThis, 'fetch').mockImplementation(Object.assign(() => { calls++; throw new Error('External fetch forbidden'); }, { preconnect() {} })); });
afterAll(() => { fetchSpy.mockRestore(); expect(calls).toBe(0); });

const stamps = ['sinaxis-discovery', 'extract-atoms', 'managed-reextract', 'stale-atoms', 'life-chronicle',
  'generation-restore', 'synthesis-resynthesis', 'synthesis-cleanup'];
for (const { backend, databaseUrl } of pathologicalArms()) {
  describe(`Sinaxis Discovery retirement hold (${backend})`, () => {
    for (const days of [30, 3]) test(`preview and delete preserve all stamps inside ${days} days, then expire`, async () => {
      const { engine, close } = await isolatedSharedSkillsEngine(databaseUrl);
      try {
        if (days === 3) await engine.setConfig('generations.restore_days', '3');
        for (const stamp of [...stamps, 'unregistered']) {
          const slug = `hold/${stamp}`;
          await engine.putPage(slug, { type: 'note', title: stamp, compiled_truth: 'Synthetic retained content.', frontmatter: { retired_by: stamp } });
          await engine.softDeletePage(slug);
        }
        await engine.executeRaw("UPDATE pages SET deleted_at=now()-interval '1 hour' * $1", [days === 30 ? 96 : 71]);
        // A one-hour sweep exposes the minimum three-day hold independently of the normal 72-hour grace.
        expect((await engine.purgeDeletedPages(1, { dryRun: true })).slugs).toEqual(['hold/unregistered']);
        expect((await engine.purgeDeletedPages(1)).slugs).toEqual(['hold/unregistered']);
        expect((await engine.executeRaw('SELECT slug FROM pages ORDER BY slug')).map(r => r.slug)).toEqual(stamps.map(s => `hold/${s}`).sort());
        await engine.executeRaw("UPDATE pages SET deleted_at=now()-interval '1 hour' * $1", [days * 24 + 1]);
        expect((await engine.purgeDeletedPages(1, { dryRun: true })).slugs.sort()).toEqual(stamps.map(s => `hold/${s}`).sort());
        expect((await engine.purgeDeletedPages(1)).slugs.sort()).toEqual(stamps.map(s => `hold/${s}`).sort());
      } finally { await close(); }
    }, 120_000);
  });
}
