// Protects the unpriced ledger branch even after every reservation has settled.
import { expect, test } from 'bun:test';
import { embedCostCap } from '../src/core/embed-cost-cap.ts';
import { getCurrentBudgetTracker } from '../src/core/ai/gateway.ts';

test('an unpriced served model reports uncertainty under a derived cap with no pending calls', async () => {
  const cap = embedCostCap(1, 'derived')!;
  await cap.metered(async () => {
    const tracker = getCurrentBudgetTracker()!;
    const reservation = tracker.reserve({ modelId: 'openai:text-embedding-3-large', estimatedInputTokens: 100, maxOutputTokens: 0, kind: 'embed' });
    tracker.record({ modelId: 'unknown:synthetic-unpriced', inputTokens: 100, reservation, kind: 'embed' });
    expect(tracker.snapshot().models).toEqual([expect.objectContaining({ cost_basis: 'measured', cost_usd: null })]);
    expect(tracker.totalReserved).toBe(0);
    expect(cap.usage()).toEqual({ spent_usd: tracker.totalSpent, spent_estimated: true });
  });
  expect(cap.hit()).toBeNull();
});
