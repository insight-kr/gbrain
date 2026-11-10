/**
 * Internal Sinaxis Discovery generation entry. Import this exact module from the pinned checkout;
 * it is deliberately not a package export or a general CLI/MCP dispatch surface.
 *
 * This layer records projection receipts, not Resource or Source authority. The caller must supply
 * an authoritative control fence held until the complete database operation commits, including reads.
 * It must deny revoked sources/resources and expired leases; a cached snapshot is not that fence.
 * No schema initialization, background work or provider call is performed here.
 */
import { AsyncLocalStorage } from 'node:async_hooks';
import { repairRunner } from '../repair/registry.ts';
import { resolveRepairScope } from '../repair/core.ts';
import type { BrainEngine } from '../engine.ts';
import type { GenerationAdapter, GenerationKey, ProposedItem } from '../generations/contract.ts';
import { registerGenerationAdapter } from '../generations/adapters/index.ts';
import { importPageItem, markdownItem, pageItemRow, preimageIdentity, preimageMarkdown, readPageRows } from '../generations/adapters/page.ts';
import { itemSetHash, lockGeneration, publishGeneration, readLiveGeneration } from '../generations/publish.ts';
import { retireDerivedPagesInTransaction } from '../persistence/derived-retirement-hold.ts';
import { maintenanceTransaction } from '../persistence/attribution.ts';
import { scrubGenerationLedger } from '../generations/ledger-scrub.ts';
import { digest } from '../persistence/digest.ts';
import { serializeMarkdown } from '../markdown.ts';
import { managedPersistenceEnabled } from '../persistence/ownership.ts';

export const DISCOVERY_KIND = 'sinaxis_discovery';
export interface DiscoveryBinding { revision: string; profileDigest: string; epoch: number; lease: string }
export interface DiscoveryState extends DiscoveryBinding { enabled: boolean }
export interface DiscoveryAuthority {
  /** Hold the current Source/Resource control fence, with no authority-changing gap before run commits. */
  withFence<T>(key: GenerationKey, run: () => Promise<T>): Promise<T>;
  /** Re-read current authority, not the producer's supplied snapshot. The fence above remains held. */
  read(tx: BrainEngine, key: GenerationKey): Promise<DiscoveryState>;
}
const restoreContext = new AsyncLocalStorage<{ key: GenerationKey; context: Context }>();
interface Context { binding: DiscoveryBinding; authority: DiscoveryAuthority }
export interface DiscoveryItem { segmentId: string; text: string }
export interface DiscoveryPublish {
  key: GenerationKey; binding: DiscoveryBinding; expectedGeneration: number; items: readonly DiscoveryItem[]; partial?: boolean;
}
export interface DiscoveryReceipt {
  generationId: number; generation: number; itemSetHash: string; complete: true;
  revision: string; profileDigest: string; items: Array<{ key: string; identityHash: string }>;
}
const refuse = () => new Error('Discovery authority, identity or receipt does not match; nothing is served.');
function validate(key: GenerationKey, b: DiscoveryBinding): void {
  if (key.kind !== DISCOVERY_KIND || !key.sourceId || !key.ownerKey.startsWith('resource:') || !b.revision || !b.lease
    || !/^[a-f0-9]{64}$/.test(b.profileDigest) || !Number.isSafeInteger(b.epoch) || b.epoch < 0) throw refuse();
}
const same = (a: DiscoveryBinding, b: DiscoveryBinding) =>
  a.revision === b.revision && a.profileDigest === b.profileDigest && a.epoch === b.epoch && a.lease === b.lease;
const pin = (b: DiscoveryBinding) => digest(['sinaxis-discovery-input-v1', b.revision, b.profileDigest, b.epoch, b.lease]);
export function discoveryKey(sourceId: string, resourceId: string): GenerationKey {
  if (!sourceId || !resourceId) throw refuse();
  return { kind: DISCOVERY_KIND, sourceId, ownerKey: `resource:${resourceId}` };
}
const prefix = (key: GenerationKey) => `sinaxis-discovery/${digest([key.sourceId, key.ownerKey])}/`;
async function check(tx: BrainEngine, key: GenerationKey, ctx: Context, enabled = true) {
  const current = await ctx.authority.read(tx, key);
  if (current.enabled !== enabled || !same(current, ctx.binding)) throw refuse();
  return { pin: pin(current), size: 0 };
}
export const discoveryAdapter: GenerationAdapter<Context> = {
  kind: DISCOVERY_KIND,
  retiredBy: 'sinaxis-discovery',
  modelProduced: false,
  // A stable anchor serializes publication/read/purge before taking the native generation lock.
  lockKeys: key => [{ sourceId: key.sourceId, slug: `${prefix(key)}anchor` }],
  readInput: async (tx, key, ctx) => {
    const scoped = restoreContext.getStore();
    const context = ctx ?? (scoped && digest(scoped.key) === digest(key) ? scoped.context : undefined);
    if (!context) throw refuse();
    return check(tx, key, context);
  },
  async readItems(tx, key, keys) {
    const rows = await readPageRows(tx, key.sourceId,
      "p.slug=ANY($2::text[]) OR (p.deleted_at IS NULL AND p.frontmatter->>'sinaxis_discovery_owner'=$3)",
      [[...keys], key.ownerKey]);
    return new Map(rows.map(row => [row.slug, pageItemRow(row,
      row.slug.startsWith(prefix(key)) && row.frontmatter?.sinaxis_discovery_owner === key.ownerKey)]));
  },
  legacyEvidence: async () => new Set<string>(),
  async publishItems(tx, key, items) {
    for (const item of items) {
      // Native import is explicit noEmbed; the product owns embedding and sanitized projection admission.
      await importPageItem(tx, key.sourceId, item.key, item.content);
    }
  },
  completeItems: async (tx, key, _items, ctx) => { if (!ctx) throw refuse(); await check(tx, key, ctx); },
  retireItems: (tx, key, keys, stamp) => retireDerivedPagesInTransaction(tx, key.sourceId, keys, stamp),
  restoreItem: (tx, key, slug, preimage) => importPageItem(tx, key.sourceId, slug, preimageMarkdown(preimage)),
  preimageIdentity,
};
registerGenerationAdapter(discoveryAdapter);

/** Database-only projection entry. Managed file-backed brains are deliberately refused, not bypassed. */
export function createDiscoveryEntry(engine: BrainEngine, authority: DiscoveryAuthority) {
  async function fenced<T>(key: GenerationKey, b: DiscoveryBinding, run: () => Promise<T>, enabled = true): Promise<T> {
    validate(key, b);
    if (await managedPersistenceEnabled(engine)) throw new Error('Discovery requires its dedicated unmanaged projection database.');
    return authority.withFence(key, async () => { await check(engine, key, { binding: b, authority }, enabled); return run(); });
  }
  return {
    async publish(request: DiscoveryPublish) {
      const { key, binding, expectedGeneration } = request;
      if (!Number.isSafeInteger(expectedGeneration) || expectedGeneration < 0) throw refuse();
      const ids = new Set<string>();
      const items: ProposedItem[] = request.items.map(item => {
        if (!item.segmentId || ids.has(item.segmentId) || typeof item.text !== 'string') throw refuse();
        ids.add(item.segmentId);
        const slug = `${prefix(key)}${digest(item.segmentId)}`;
        return markdownItem(slug, serializeMarkdown({
          sinaxis_discovery_owner: key.ownerKey, revision: binding.revision, profile_digest: binding.profileDigest, authority_pin: pin(binding), discovery_complete: request.partial !== true, embed_skip: true,
        }, item.text, '', { type: 'note', title: item.segmentId, tags: [] }));
      });
      return fenced(key, binding, () => publishGeneration(engine, discoveryAdapter, {
        key, items, input: { pin: pin(binding), size: 0 }, expectedGeneration, partial: request.partial,
        context: { binding, authority }, note: { revision: binding.revision, profileDigest: binding.profileDigest },
      }));
    },
    async read(key: GenerationKey, binding: DiscoveryBinding): Promise<DiscoveryReceipt | null> {
      return fenced(key, binding, () => maintenanceTransaction(engine, async tx => {
        await lockGeneration(tx, discoveryAdapter, key, []);
        await check(tx, key, { binding, authority });
        const live = await readLiveGeneration(tx, key);
        if (!live) return null;
        if (live.inputPin !== pin(binding)) throw refuse();
        const [row] = await tx.executeRaw<{ item_set_hash: string; completeness: Record<string, unknown> }>(
          'SELECT item_set_hash, completeness FROM derived_generations WHERE id=$1', [live.id]);
        if (row?.completeness.complete !== true) return null;
        if (!row.completeness.restore && (row.completeness.revision !== binding.revision || row.completeness.profileDigest !== binding.profileDigest)) throw refuse();
        const items = [...live.items].map(([key, identityHash]) => ({ key, identityHash }));
        if (itemSetHash(items) !== row.item_set_hash) throw refuse();
        const rows = await discoveryAdapter.readItems(tx, key, items.map(i => i.key));
        if (items.some(i => !rows.get(i.key)?.live || rows.get(i.key)?.identityHash !== i.identityHash)) throw refuse();
        const pages = await readPageRows(tx, key.sourceId, 'p.slug=ANY($2::text[])', [items.map(i => i.key)]);
        if (pages.some(p => p.frontmatter?.revision !== binding.revision || p.frontmatter?.profile_digest !== binding.profileDigest)) throw refuse();
        return { generationId: live.id, generation: live.generation, itemSetHash: row.item_set_hash,
          complete: true, revision: binding.revision, profileDigest: binding.profileDigest, items };
      }));
    },
    /** Restorable retirement is not withdrawal: the caller still holds active authority. */
    async retire(key: GenerationKey, binding: DiscoveryBinding, expectedGeneration: number) {
      return fenced(key, binding, () => maintenanceTransaction(engine, async tx => {
        await lockGeneration(tx, discoveryAdapter, key, []);
        await check(tx, key, { binding, authority });
        const live = await readLiveGeneration(tx, key);
        return publishGeneration(tx, discoveryAdapter, {
          key, items: [], expectedGeneration, input: { pin: pin(binding), size: 0 },
          context: { binding, authority }, approved: { retire: [...(live?.items.keys() ?? [])], retiredBy: 'sinaxis-discovery' },
          note: { revision: binding.revision, profileDigest: binding.profileDigest },
        });
      }));
    },
    /** Native preview-bound restore, scoped to this authority fence; raw repair has no control context. */
    async restore(key: GenerationKey, binding: DiscoveryBinding, expectedPreview?: string) {
      return fenced(key, binding, () => restoreContext.run({ key, context: { binding, authority } }, async () => {
        await check(engine, key, { binding, authority });
        return (await repairRunner(engine, { apply: expectedPreview !== undefined, noEmbed: true,
          logger: { info() {}, warn() {}, error() {} } })).run('generations', await resolveRepairScope(engine, key.sourceId),
          { explicit: true, sourceFlag: key.sourceId, generations: { kind: DISCOVERY_KIND, mode: 'restore', key: key.ownerKey },
            ...(expectedPreview ? { expect: expectedPreview } : {}) });
      }));
    },
    /** Explicit irreversible cleanup only AFTER current control authority has disabled serving. */
    async purge(key: GenerationKey, binding: DiscoveryState) {
      return fenced(key, binding, () => maintenanceTransaction(engine, async tx => {
        await lockGeneration(tx, discoveryAdapter, key, []);
        if (binding.enabled !== false) throw refuse();
        await check(tx, key, { binding, authority }, false);
        const rows = await readPageRows(tx, key.sourceId, "p.frontmatter->>'sinaxis_discovery_owner'=$2", [key.ownerKey]);
        for (const row of rows) {
          if (!row.slug.startsWith(prefix(key))) throw refuse();
          await scrubGenerationLedger(tx, key.sourceId, { slug: row.slug });
          await tx.deletePage(row.slug, { sourceId: key.sourceId });
        }
        return { purged: rows.length };
      }), false);
    },
  };
}
