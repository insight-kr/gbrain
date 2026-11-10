/**
 * Protects native Discovery receipts and atomic replacement, fencing, retirement, restore and withdrawal.
 * A missing entry, a stale publisher or a partial receipt exposed as complete fails these cases.
 * Existing synthetic-kind tests do not exercise Discovery identity/profile/authority checks. No new test seam.
 */
import { afterAll, beforeAll, describe, expect, spyOn, test } from 'bun:test';
import { createDiscoveryEntry, discoveryKey, type DiscoveryAuthority, type DiscoveryBinding } from '../../src/core/sinaxis-discovery/entry.ts';
import { isolatedSharedSkillsEngine } from '../helpers/shared-skills-engine.ts';
import { pathologicalArms } from './harness.ts';
import { installFaultHook } from '../../src/core/persistence/fault-points.ts';

let calls = 0;
let fetchSpy: ReturnType<typeof spyOn>;
beforeAll(() => { fetchSpy = spyOn(globalThis, 'fetch').mockImplementation(Object.assign(() => { calls++; throw new Error('External fetch forbidden'); }, { preconnect() {} })); });
afterAll(() => { fetchSpy.mockRestore(); expect(calls).toBe(0); });

const binding: DiscoveryBinding = { revision: 'revision-1', profileDigest: 'a'.repeat(64), epoch: 1, lease: 'lease-1' };
const key = discoveryKey('default', 'resource-1');
for (const { backend, databaseUrl } of pathologicalArms()) {
  describe(`Sinaxis Discovery generations (${backend})`, () => {
    async function fixture(run: (f: Awaited<ReturnType<typeof make>>) => Promise<void>) {
      const f = await make();
      try { await run(f); } finally { installFaultHook(undefined); await f.close(); }
    }
    async function make() {
      const { engine, close } = await isolatedSharedSkillsEngine(databaseUrl);
      let current = { ...binding, enabled: true };
      const authority: DiscoveryAuthority = {
        // Synthetic authority only: the product must hold its own control fence across this scope.
        withFence: async (_key, run) => run(),
        read: async () => ({ ...current }),
      };
      const entry = createDiscoveryEntry(engine, authority);
      const request = (names = ['a', 'b'], expectedGeneration = 0) => ({
        key, binding: { ...current }, expectedGeneration,
        items: names.map(name => ({ segmentId: name, text: `Synthetic segment ${name}.` })),
      });
      return { engine, close, entry, request, move: (patch: Partial<typeof current>) => { current = { ...current, ...patch }; } };
    }
    test('publishes a complete native receipt; verifies item-set, profile and revision; rejects mismatches', () => fixture(async f => {
      expect(await f.entry.publish(f.request())).toMatchObject({ state: 'published', generation: 1 });
      const receipt = await f.entry.read(key, binding);
      expect(receipt).toMatchObject({ generation: 1, revision: binding.revision, profileDigest: binding.profileDigest, complete: true });
      expect(receipt?.items).toHaveLength(2);
      expect(receipt?.itemSetHash).toMatch(/^[a-f0-9]{64}$/);
      await expect(f.entry.read(key, { ...binding, revision: 'wrong' })).rejects.toThrow();
      await expect(f.entry.read(key, { ...binding, profileDigest: 'b'.repeat(64) })).rejects.toThrow();
      expect((await f.engine.executeRaw("SELECT item_set_hash, completeness FROM derived_generations WHERE state='live'"))[0]).toMatchObject({
        item_set_hash: receipt?.itemSetHash, completeness: expect.objectContaining({ complete: true }),
      });
    }), 120_000);
    test('concurrent publishers admit one successor and fence the other without lost items', () => fixture(async f => {
      await f.entry.publish(f.request());
      const results = await Promise.allSettled([f.entry.publish(f.request(['c'], 1)), f.entry.publish(f.request(['d'], 1))]);
      expect(results.filter(r => r.status === 'fulfilled')).toHaveLength(1);
      expect(results.filter(r => r.status === 'rejected')).toHaveLength(1);
      expect((await f.entry.read(key, binding))?.generation).toBe(2);
    }), 120_000);
    test('lease/epoch loss fences a queued producer; disabled authority blocks read and write', () => fixture(async f => {
      const stale = f.request(); f.move({ lease: 'lease-2', epoch: 2 });
      await expect(f.entry.publish(stale)).rejects.toThrow();
      expect(await f.engine.executeRaw('SELECT id FROM derived_generations')).toHaveLength(0);
      f.move({ enabled: false });
      await expect(f.entry.publish(f.request())).rejects.toThrow();
      await expect(f.entry.read(key, binding)).rejects.toThrow();
    }), 120_000);
    test('held input writes nothing; partial publication is never a serving receipt', () => fixture(async f => {
      await f.entry.publish(f.request());
      expect(await f.entry.publish(f.request([], 1))).toMatchObject({ state: 'held', reason: 'no_proof' });
      expect((await f.entry.read(key, binding))?.generation).toBe(1);
      expect(await f.entry.publish({ ...f.request(['c'], 1), partial: true })).toMatchObject({ state: 'published', generation: 2 });
      expect(await f.entry.read(key, binding)).toBeNull();
      expect((await f.engine.executeRaw('SELECT slug FROM pages WHERE deleted_at IS NULL'))).toHaveLength(3);
      expect(await f.entry.publish(f.request(['a', 'b', 'c'], 2))).toMatchObject({ state: 'published', generation: 3 });
      expect((await f.entry.read(key, binding))?.items).toHaveLength(3);
    }), 120_000);
    test('retire keeps restorable preimages; native repair restores the previous generation', () => fixture(async f => {
      await f.entry.publish(f.request());
      expect(await f.entry.retire(key, binding, 1)).toMatchObject({ state: 'published', generation: 2 });
      expect((await f.engine.executeRaw("SELECT frontmatter->>'retired_by' AS stamp FROM pages"))).toEqual([
        { stamp: 'sinaxis-discovery' }, { stamp: 'sinaxis-discovery' },
      ]);
      const run = (hash?: string) => f.entry.restore(key, binding, hash);
      const preview = await run(); expect(preview.listing?.[0]?.class).toBe('restore');
      expect((await run(preview.preview_hash)).outcomes).toEqual({ restored: 1 });
      expect((await f.engine.executeRaw('SELECT slug FROM pages WHERE deleted_at IS NULL'))).toHaveLength(2);
      expect((await f.entry.read(key, binding))?.items).toHaveLength(2);
    }), 120_000);
    test('withdrawal denies serving first, then explicit purge removes pages and retained preimages', () => fixture(async f => {
      await f.entry.publish(f.request()); await f.entry.publish(f.request(['c'], 1));
      await expect(f.entry.purge(key, { ...binding, enabled: true })).rejects.toThrow();
      f.move({ enabled: false });
      await expect(f.entry.read(key, binding)).rejects.toThrow();
      await f.entry.purge(key, { ...binding, enabled: false });
      expect(await f.engine.executeRaw('SELECT id FROM pages')).toHaveLength(0);
      expect(await f.engine.executeRaw('SELECT preimage FROM derived_generation_items WHERE preimage IS NOT NULL')).toHaveLength(0);
      await expect(f.entry.publish(f.request())).rejects.toThrow();
    }), 120_000);
    test('identical replay cannot bypass withdrawal and a new epoch cannot serve an old receipt', () => fixture(async f => {
      const original = f.request(); await f.entry.publish(original);
      f.move({ enabled: false });
      await expect(f.entry.publish(original)).rejects.toThrow();
      f.move({ enabled: true, epoch: 2 });
      await expect(f.entry.read(key, { ...binding, epoch: 2 })).rejects.toThrow();
      expect(await f.entry.publish(f.request(['a', 'b'], 1))).toMatchObject({ state: 'published', generation: 2 });
    }), 120_000);
    test('a failed transaction leaves no generation; a new entry retries from durable state', () => fixture(async f => {
      installFaultHook(point => { if (point === 'generation:published') throw new Error('synthetic crash'); });
      await expect(f.entry.publish(f.request())).rejects.toThrow('synthetic crash');
      installFaultHook(undefined);
      expect(await f.engine.executeRaw('SELECT id FROM pages')).toHaveLength(0);
      expect(await f.engine.executeRaw("SELECT id FROM derived_generations WHERE state='live'")).toHaveLength(0);
      expect(await f.entry.publish(f.request())).toMatchObject({ state: 'published', generation: 1 });
    }), 120_000);
  });
}
