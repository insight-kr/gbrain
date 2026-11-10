/**
 * Review regressions: withdrawal scrubs never-published payloads and authority fences surround every DB operation.
 * These cases fail if a revoked producer retains proposals or if entry reads/commits escape the control fence.
 */
import { afterAll, beforeAll, describe, expect, spyOn, test } from 'bun:test';
import { createDiscoveryEntry, discoveryKey, type DiscoveryAuthority, type DiscoveryBinding } from '../../src/core/sinaxis-discovery/entry.ts';
import type { BrainEngine } from '../../src/core/engine.ts';
import { repairRunner } from '../../src/core/repair/registry.ts';
import { resolveRepairScope } from '../../src/core/repair/core.ts';
import { isolatedSharedSkillsEngine } from '../helpers/shared-skills-engine.ts';
import { pathologicalArms } from './harness.ts';
import { installFaultHook } from '../../src/core/persistence/fault-points.ts';

let calls = 0;
let fetchSpy: ReturnType<typeof spyOn>;
beforeAll(() => { fetchSpy = spyOn(globalThis, 'fetch').mockImplementation(Object.assign(() => { calls++; throw new Error('External fetch forbidden'); }, { preconnect() {} })); });
afterAll(() => { fetchSpy.mockRestore(); expect(calls).toBe(0); });
const binding: DiscoveryBinding = { revision: 'review-revision', profileDigest: 'c'.repeat(64), epoch: 1, lease: 'review-lease' };
const key = discoveryKey('default', 'review-resource');
for (const { backend, databaseUrl } of pathologicalArms()) {
  describe(`Sinaxis Discovery review regressions (${backend})`, () => {
    async function fixture(run: (f: Awaited<ReturnType<typeof make>>) => Promise<void>) {
      const f = await make(); try { await run(f); } finally { installFaultHook(undefined); await f.close(); }
    }
    async function make() {
      const { engine, close } = await isolatedSharedSkillsEngine(databaseUrl);
      let current = { ...binding, enabled: true };
      let depth = 0, dbCalls = 0, commits = 0, reads = 0, denyAt = Infinity, rejectFence = false, enforce = false;
      const guarded = (db: BrainEngine): BrainEngine => new Proxy(db, {
        get(target, prop) {
          const value = Reflect.get(target, prop);
          if (typeof value !== 'function') return value;
          return (...args: unknown[]) => {
            if (enforce) expect(depth).toBeGreaterThan(0); dbCalls++;
            if (prop === 'transaction') {
              const fn = args[0] as (tx: BrainEngine) => Promise<unknown>;
              return target.transaction(tx => fn(guarded(tx))).then(result => {
                if (enforce) expect(depth).toBeGreaterThan(0); commits++; return result;
              });
            }
            return Reflect.apply(value, target, args);
          };
        },
      });
      const authority: DiscoveryAuthority = {
        withFence: async (_key, run) => {
          if (rejectFence) throw new Error('Synthetic fence refused');
          depth++; try { return await run(); } finally { depth--; }
        },
        read: async () => { expect(depth).toBeGreaterThan(0); reads++; return { ...current, enabled: reads >= denyAt ? false : current.enabled }; },
      };
      const entry = createDiscoveryEntry(guarded(engine), authority);
      const request = (names = ['a'], expectedGeneration = 0) => ({
        key, binding, expectedGeneration, items: names.map(segmentId => ({ segmentId, text: `Synthetic secret ${segmentId}` })),
      });
      return { engine, close, entry, request,
        move: (patch: Partial<typeof current>) => { current = { ...current, ...patch }; },
        deny: (at: number) => { reads = 0; denyAt = at; },
        reject: () => { rejectFence = true; }, enforce: () => { enforce = true; },
        stats: () => ({ dbCalls, commits, depth }),
      };
    }
    test('purge scrubs held proposals without pages plus partial/preimage payloads, scoped to one resource', () => fixture(async f => {
      await f.entry.publish(f.request());
      await f.entry.publish(f.request(['b'], 1));
      await f.entry.publish({ ...f.request(['c'], 2), partial: true });
      expect(await f.entry.publish(f.request(Array.from({ length: 65 }, (_, i) => `held-${i}`), 3))).toMatchObject({ state: 'held' });
      const other = discoveryKey('default', 'untouched');
      await f.entry.publish({ ...f.request(['other']), key: other });
      await f.entry.publish({ ...f.request(Array.from({ length: 65 }, (_, i) => `other-${i}`), 1), key: other });
      expect(await f.engine.executeRaw('SELECT proposal FROM derived_generations WHERE owner_key=$1 AND proposal IS NOT NULL', [key.ownerKey])).toHaveLength(1);
      f.move({ enabled: false });
      const result = await f.entry.purge(key, { ...binding, enabled: false });
      expect(await f.engine.executeRaw('SELECT proposal FROM derived_generations WHERE owner_key=$1 AND proposal IS NOT NULL', [key.ownerKey])).toHaveLength(0);
      expect(await f.engine.executeRaw('SELECT i.preimage FROM derived_generation_items i JOIN derived_generations g ON g.id=i.generation_id WHERE g.owner_key=$1 AND i.preimage IS NOT NULL', [key.ownerKey])).toHaveLength(0);
      expect(await f.engine.executeRaw("SELECT slug FROM pages WHERE frontmatter->>'sinaxis_discovery_owner'=$1", [key.ownerKey])).toHaveLength(0);
      expect(await f.engine.executeRaw("SELECT slug FROM pages WHERE frontmatter->>'sinaxis_discovery_owner'=$1", [other.ownerKey])).toHaveLength(1);
      expect(await f.engine.executeRaw('SELECT proposal FROM derived_generations WHERE owner_key=$1 AND proposal IS NOT NULL', [other.ownerKey])).toHaveLength(1);
      expect(result).toMatchObject({ purged: 3, derived_generations: 1 });
      expect(result.derived_generation_items).toBeGreaterThan(0);
    }), 120_000);
    test('purge removes a never-published held proposal even when this key has no page rows', () => fixture(async f => {
      expect(await f.entry.publish(f.request(Array.from({ length: 65 }, (_, i) => `held-only-${i}`)))).toMatchObject({ state: 'held' });
      expect(await f.engine.executeRaw('SELECT id FROM pages')).toHaveLength(0);
      f.move({ enabled: false });
      expect(await f.entry.purge(key, { ...binding, enabled: false })).toMatchObject({ purged: 0, ledger_keys: 65, derived_generations: 1 });
      expect(await f.engine.executeRaw('SELECT proposal FROM derived_generations WHERE proposal IS NOT NULL')).toHaveLength(0);
    }), 120_000);
    test('purge also scrubs never-published attempts_exhausted proposals from terminal failures', () => fixture(async f => {
      installFaultHook(point => { if (point === 'generation:published') throw new Error('Synthetic terminal failure'); });
      for (let i = 0; i < 3; i++) await expect(f.entry.publish(f.request())).rejects.toThrow('Synthetic terminal failure');
      installFaultHook(undefined);
      expect(await f.entry.publish(f.request())).toMatchObject({ state: 'held', reason: 'attempts_exhausted' });
      expect(await f.engine.executeRaw('SELECT id FROM pages')).toHaveLength(0);
      f.move({ enabled: false });
      expect(await f.entry.purge(key, { ...binding, enabled: false })).toMatchObject({ purged: 0, derived_generations: 1 });
      expect(await f.engine.executeRaw('SELECT proposal FROM derived_generations WHERE proposal IS NOT NULL')).toHaveLength(0);
    }), 120_000);
    test('receipt refuses content tampering rather than serving stale identity evidence', () => fixture(async f => {
      await f.entry.publish(f.request());
      await f.engine.executeRaw("UPDATE pages SET compiled_truth='Synthetic tampering'");
      await expect(f.entry.read(key, binding)).rejects.toThrow();
    }), 120_000);
    test('foreign-prefix ownership contamination refuses the entire purge without partial deletion', () => fixture(async f => {
      await f.entry.publish(f.request(['a', 'b']));
      await f.engine.executeRaw("UPDATE pages SET slug='outside-discovery-prefix' WHERE title='a'");
      const before = await f.engine.executeRaw('SELECT slug FROM pages ORDER BY slug');
      f.move({ enabled: false });
      await expect(f.entry.purge(key, { ...binding, enabled: false })).rejects.toThrow();
      expect(await f.engine.executeRaw('SELECT slug FROM pages ORDER BY slug')).toEqual(before);
    }), 120_000);
    test('repeated authority loss inside publish is fenced, never attempted or retained as raw proposals', () => fixture(async f => {
      for (let i = 0; i < 4; i++) {
        f.deny(2);
        await expect(f.entry.publish(f.request())).rejects.toMatchObject({ code: 'generation_fenced' });
      }
      expect(await f.engine.executeRaw('SELECT id FROM derived_generations')).toHaveLength(0);
      expect(await f.engine.executeRaw('SELECT * FROM derived_generation_attempts')).toHaveLength(0);
      expect(await f.engine.executeRaw('SELECT id FROM pages')).toHaveLength(0);
    }), 120_000);
    test('authority moving after writes rolls back pages and ledger without counting an attempt', () => fixture(async f => {
      f.deny(3);
      await expect(f.entry.publish(f.request())).rejects.toMatchObject({ code: 'generation_fenced' });
      expect(await f.engine.executeRaw('SELECT id FROM pages')).toHaveLength(0);
      expect(await f.engine.executeRaw('SELECT id FROM derived_generations')).toHaveLength(0);
      expect(await f.engine.executeRaw('SELECT * FROM derived_generation_attempts')).toHaveLength(0);
    }), 120_000);
    test('all entry reads, transactions and their commit completion remain inside withFence', () => fixture(async f => {
      f.enforce();
      await f.entry.publish(f.request());
      expect((await f.entry.read(key, binding))?.items).toHaveLength(1);
      await f.entry.retire(key, binding, 1);
      const preview = await f.entry.restore(key, binding);
      await f.entry.restore(key, binding, preview.preview_hash);
      f.move({ enabled: false });
      await f.entry.purge(key, { ...binding, enabled: false });
      expect(f.stats().dbCalls).toBeGreaterThan(0);
      expect(f.stats().commits).toBeGreaterThan(0);
      expect(f.stats().depth).toBe(0);
    }), 120_000);
    test('a refusing/throwing fence performs no DB reads or writes', () => fixture(async f => {
      f.reject();
      await expect(f.entry.publish(f.request())).rejects.toThrow('Synthetic fence refused');
      await expect(f.entry.read(key, binding)).rejects.toThrow('Synthetic fence refused');
      await expect(f.entry.retire(key, binding, 0)).rejects.toThrow('Synthetic fence refused');
      await expect(f.entry.restore(key, binding)).rejects.toThrow('Synthetic fence refused');
      await expect(f.entry.purge(key, { ...binding, enabled: false })).rejects.toThrow('Synthetic fence refused');
      expect(f.stats().dbCalls).toBe(0);
      expect(await f.engine.executeRaw('SELECT id FROM pages')).toHaveLength(0);
    }), 120_000);
    test('raw native restore without scoped control context fails closed and rolls back', () => fixture(async f => {
      await f.entry.publish(f.request());
      await f.entry.retire(key, binding, 1);
      const preview = await f.entry.restore(key, binding);
      const before = await f.engine.executeRaw('SELECT id, state FROM derived_generations ORDER BY id');
      await expect((await repairRunner(f.engine, { apply: true, noEmbed: true, logger: { info() {}, warn() {}, error() {} } }))
        .run('generations', await resolveRepairScope(f.engine, key.sourceId),
          { explicit: true, sourceFlag: key.sourceId, generations: { kind: key.kind, mode: 'restore', key: key.ownerKey }, expect: preview.preview_hash })).rejects.toThrow();
      expect(await f.engine.executeRaw('SELECT id, state FROM derived_generations ORDER BY id')).toEqual(before);
      expect(await f.engine.executeRaw('SELECT id FROM pages WHERE deleted_at IS NULL')).toHaveLength(0);
    }), 120_000);
    test('retired receipt reads as absent and a restored receipt serves again', () => fixture(async f => {
      await f.entry.publish(f.request()); await f.entry.retire(key, binding, 1);
      expect(await f.entry.read(key, binding)).toBeNull();
      const preview = await f.entry.restore(key, binding);
      await f.entry.restore(key, binding, preview.preview_hash);
      expect((await f.entry.read(key, binding))?.items).toHaveLength(1);
    }), 120_000);
    test('oversized segment refuses before publication without attempts or raw ledger retention', () => fixture(async f => {
      await expect(f.entry.publish({ ...f.request(), items: [{ segmentId: 'oversized', text: '💎'.repeat(256 * 1024 + 1) }] })).rejects.toThrow('Discovery segment exceeds');
      expect(await f.engine.executeRaw('SELECT * FROM derived_generation_attempts')).toHaveLength(0);
      expect(await f.engine.executeRaw('SELECT id FROM derived_generations')).toHaveLength(0);
      expect(f.stats().dbCalls).toBe(0);
    }), 120_000);
  });
}
