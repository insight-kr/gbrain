// Protects native REST cardinality, validation, settlement and refusal behavior.
// Wrong transport/usage/limit branches fail; legacy tests do not cover native Google.
// Uses the existing recipe seam and fetch double; adds no production-only test hook.
import { afterEach, beforeEach, expect, test } from 'bun:test';
import { configureGateway, embedMultimodal, embedMultimodalSafe, withBudgetTracker, resetGateway } from '../../src/core/ai/gateway.ts';
import { __setTestRecipesForTests } from '../../src/core/ai/recipes/index.ts';
import { google } from '../../src/core/ai/recipes/google.ts';
import { AIConfigError, AITransientError } from '../../src/core/ai/errors.ts';

import { BudgetTracker } from '../../src/core/budget/budget-tracker.ts';
import { withAIInvocationGuard, sdkInvocationUsage } from '../../src/core/ai/invocation-guard.ts';
import { resolveSchemaEmbeddingDim } from '../../src/core/embedding-dim-check.ts';

const originalFetch = globalThis.fetch;
let calls: Array<{ url: string; init: RequestInit; body: any }>;
let response: () => Response;
function configure() {
  configureGateway({ embedding_model: 'google-fixture:gemini-embedding-2',
    env: { GOOGLE_GENERATIVE_AI_API_KEY: 'synthetic', GOOGLE_GENERATIVE_AI_BASE_URL: 'http://127.0.0.1:9/proxy' } });
}
beforeEach(() => {
  calls = [];
  __setTestRecipesForTests([{ ...google, id: 'google-fixture', touchpoints: { ...google.touchpoints,
    embedding: { ...google.touchpoints.embedding!, supports_multimodal: true, multimodal_models: ['gemini-embedding-2'] } } }]);
  configure();
  response = () => Response.json({ embedding: { values: Array(768).fill(calls.length) }, usageMetadata: { promptTokenCount: 23 } });
  globalThis.fetch = (async (url, init) => {
    calls.push({ url: String(url), init: init!, body: JSON.parse(init!.body as string) });
    return response();
  }) as typeof fetch;
});
afterEach(() => { globalThis.fetch = originalFetch; __setTestRecipesForTests([]); resetGateway(); });

test('Google sends one Content per input and reads singular REST vectors in order', async () => {
  const vectors = await embedMultimodal([
    { kind: 'content', parts: [{ kind: 'text', text: 'caption' }, { kind: 'text', text: 'context' }] },
    { kind: 'text', text: 'second' },
    { kind: 'text', text: 'third' },
  ]);
  expect(vectors.map(v => v[0])).toEqual([1, 2, 3]);
  expect(calls.map(c => c.url)).toEqual(Array(3).fill('http://127.0.0.1:9/proxy/v1beta/models/gemini-embedding-2:embedContent'));
  expect(calls[0].body).toEqual({ content: { parts: [{ text: 'caption' }, { text: 'context' }] },
    embedContentConfig: { outputDimensionality: 768, autoTruncate: false } });
  expect(calls[1].body.content.parts).toEqual([{ text: 'second' }]);
  expect(calls[2].body.content.parts).toEqual([{ text: 'third' }]);
  expect(new Headers(calls[0].init.headers).get('x-goog-api-key')).toBe('synthetic');
});

test('Google rejects plural SDK/batch responses and coerced numeric values', async () => {
  for (const payload of [
    { embeddings: [{ values: Array(768).fill(1) }] },
    { embedding: { values: Array(768).fill('1') } },
    { embedding: { values: Array(768).fill(null) } },
  ]) {
    response = () => Response.json(payload);
    await expect(embedMultimodal([{ kind: 'text', text: 'query' }])).rejects.toBeInstanceOf(AITransientError);
  }
});

test('Google keeps production capability closed and refuses file URI before HTTP', async () => {
  configureGateway({ embedding_model: 'google:gemini-embedding-2', env: {} });
  await expect(embedMultimodal([{ kind: 'text', text: 'query' }])).rejects.toBeInstanceOf(AIConfigError);
  configure();
  await expect(embedMultimodal([{ kind: 'content', parts: [{ kind: 'file_uri', uri: 'gs://private/object', mime: 'image/png' }] }])).rejects.toBeInstanceOf(AIConfigError);
  expect(calls).toEqual([]);
});


test('Google uses explicit dimensions and supports the Embedding 2 integer range at schema preflight', async () => {
  response = () => Response.json({ embedding: { values: Array(129).fill(1) } });
  const [vec] = await embedMultimodal([{ kind: 'text', text: 'query' }], { dimensions: 129 });
  expect(vec.length).toBe(129);
  expect(calls[0].body.embedContentConfig.outputDimensionality).toBe(129);
  expect(resolveSchemaEmbeddingDim({ embedding_model: 'google:gemini-embedding-2', embedding_dimensions: 129 }).ok).toBe(true);
  expect(resolveSchemaEmbeddingDim({ embedding_model: 'google:gemini-embedding-001', embedding_dimensions: 129 }).ok).toBe(false);
  calls.length = 0;
  for (const dimensions of [127, 3073, 128.5, NaN]) {
    await expect(embedMultimodal([{ kind: 'text', text: 'query' }], { dimensions })).rejects.toBeInstanceOf(AIConfigError);
  }
  expect(calls).toEqual([]);
});

test('Google invocation and ambient budgets share measured usage and a conservative missing-usage ceiling', async () => {
  for (const tokens of [23, 0, undefined]) {
    response = () => Response.json({ embedding: { values: Array(768).fill(1) },
      ...(tokens === undefined ? {} : { usageMetadata: { promptTokenCount: tokens } }) });
    const tracker = new BudgetTracker({ label: 'google-fixture' });
    const admitted: any[] = [], settled: any[] = [];
    await withBudgetTracker(tracker, () => withAIInvocationGuard(async call => {
      admitted.push(call); return { settle: async usage => { settled.push(usage); } };
    }, () => embedMultimodal([{ kind: 'text', text: 'short' }])));
    expect(admitted).toEqual([{ operation: 'gateway.multimodal', kind: 'multimodal', model: 'google-fixture:gemini-embedding-2',
      maxInputTokens: 8192, estimatedInputTokens: 8192, maxOutputTokens: 0 }]);
    expect(settled).toEqual([tokens === undefined ? null : { inputTokens: tokens, outputTokens: 0 }]);
    const row = tracker.snapshot().models[0];
    expect(row.input_tokens).toBe(tokens ?? 8192);
    expect(row.cost_basis).toBe(tokens === undefined ? 'estimated' : 'measured');
  }
});

test('Google never refunds an attempted malformed response but releases a refused admission', async () => {
  const tracker = new BudgetTracker({ label: 'google-fixture' });
  response = () => Response.json({ embedding: { values: [1] }, usageMetadata: { promptTokenCount: 23 } });
  await expect(withBudgetTracker(tracker, () => embedMultimodal([{ kind: 'text', text: 'query' }]))).rejects.toBeInstanceOf(AIConfigError);
  expect(tracker.snapshot().models[0].input_tokens).toBe(8192);
  expect(tracker.snapshot().models[0].cost_basis).toBe('estimated');
  calls.length = 0;
  const refused = new Error('refused');
  const deniedTracker = new BudgetTracker({ label: 'google-fixture' });
  await expect(withBudgetTracker(deniedTracker, () => withAIInvocationGuard(async () => { throw refused; },
    () => embedMultimodalSafe([{ kind: 'text', text: 'query' }])))).rejects.toBe(refused);
  expect(deniedTracker.snapshot().callsRecorded).toBe(0);
  expect(calls).toEqual([]);
});

test('Google usage normalizer refuses invalid counts', () => {
  for (const promptTokenCount of [-1, 1.5, '23', NaN, Infinity]) {
    expect(sdkInvocationUsage({ usageMetadata: { promptTokenCount } })).toBeNull();
  }
  expect(sdkInvocationUsage({ usageMetadata: { promptTokenCount: 0 } })).toEqual({ inputTokens: 0, outputTokens: 0 });
});

test('Google classifies HTTP failures without retaining echoed input', async () => {
  for (const status of [400, 401, 403, 429, 500]) {
    response = () => new Response('private input bytes', { status });
    let error: any;
    try { await embedMultimodal([{ kind: 'text', text: 'query' }]); } catch (err) { error = err; }
    expect(error).toBeInstanceOf(status < 500 && status !== 429 ? AIConfigError : AITransientError);
    expect(error.status).toBe(status);
    expect(String(error)).not.toContain('private input bytes');
  }
});

test('Google screens aligned vectors without losing the middle input index', async () => {
  response = () => Response.json({ embedding: { values: Array(768).fill(calls.length === 2 ? 0 : calls.length) } });
  const result = await embedMultimodalSafe(['first', 'bad', 'last'].map(text => ({ kind: 'text', text })));
  expect(result.failedIndices).toEqual([1]);
  expect(result.embeddings.map(v => v?.[0])).toEqual([1, undefined, 3]);
  expect(calls.length).toBe(3);
});

test('Google rejects unsupported MIME, too many images and oversized combined bodies before HTTP', async () => {
  const image = { kind: 'inline_data' as const, mime: 'image/png', data: 'eA==' };
  for (const parts of [
    Array(7).fill(image),
    [{ ...image, mime: 'image/svg+xml' }],
    [{ kind: 'text', text: 'x'.repeat(16 * 1024 * 1024) }],
  ]) {
    await expect(embedMultimodal([{ kind: 'content', parts } as any])).rejects.toBeInstanceOf(AIConfigError);
  }
  expect(calls).toEqual([]);
});


test('unverified PDF, audio and video remain held before any provider attempt', async () => {
  for (const mime of ['application/pdf', 'audio/mpeg', 'audio/wav', 'video/mp4', 'video/quicktime']) {
    await expect(embedMultimodal([{ kind: 'content', parts: [{ kind: 'inline_data', mime, data: 'eA==' }] }])).rejects.toBeInstanceOf(AIConfigError);
  }
  expect(calls).toEqual([]);
});

test('Google partial usage stays estimated and each retry remains admitted', async () => {
  response = () => Response.json({ embedding: { values: Array(768).fill(1) },
    ...(calls.length === 1 ? { usageMetadata: { promptTokenCount: 23 } } : {}) });
  const tracker = new BudgetTracker({ label: 'google-fixture' });
  await withBudgetTracker(tracker, () => embedMultimodal([{ kind: 'text', text: 'first' }, { kind: 'text', text: 'second' }]));
  expect(tracker.snapshot().models[0].input_tokens).toBe(16384);
  expect(tracker.snapshot().models[0].cost_basis).toBe('estimated');
});

test('Google composes caller cancellation with its timeout signal', async () => {
  const signal = AbortSignal.abort(new Error('cancelled'));
  globalThis.fetch = (async (_url: RequestInfo | URL, init?: RequestInit) => { init!.signal!.throwIfAborted(); throw new Error('unexpected'); }) as unknown as typeof fetch;
  await expect(embedMultimodal([{ kind: 'text', text: 'query' }], { abortSignal: signal })).rejects.toBeInstanceOf(AITransientError);
});

test('Google rejects non-finite vectors after Float32 conversion', async () => {
  response = () => Response.json({ embedding: { values: Array(768).fill(1e40) } });
  const result = await embedMultimodalSafe([{ kind: 'text', text: 'query' }]);
  expect(result.failedIndices).toEqual([0]);
  expect(result.embeddings).toEqual([undefined]);
});

test('Google rejects over-budget UTF-8 text and combined parts before any Content is sent', async () => {
  for (const parts of [
    [{ kind: 'text', text: '가나다라 '.repeat(40000) }],
    [{ kind: 'text', text: 'x'.repeat(8192) }],
    [{ kind: 'text', text: '가'.repeat(2731) }],
    [{ kind: 'text', text: 'x'.repeat(4096) }, { kind: 'text', text: 'y'.repeat(4096) }],
  ]) {
    await expect(embedMultimodal([{ kind: 'text', text: 'short first Content' }, { kind: 'content', parts } as any])).rejects.toThrow('shared-token');
  }
  expect(calls).toEqual([]);
});

test('Google holds images whose shared-token contribution cannot be bounded locally', async () => {
  for (const mime of ['image/png', 'image/jpeg']) {
    await expect(embedMultimodal([{ kind: 'content', parts: [
      { kind: 'text', text: 'caption' }, { kind: 'inline_data', mime, data: 'eA==' },
    ] }])).rejects.toThrow('shared-token');
    await expect(embedMultimodal([{ kind: 'image_base64', mime, data: 'eA==' }])).rejects.toThrow('shared-token');
  }
  expect(calls).toEqual([]);
});

test('Google refuses ceiling-reaching usage even for a valid short-input vector', async () => {
  for (const promptTokenCount of [8192, 9000, '8192', -1, 1.5]) {
    response = () => Response.json({ embedding: { values: Array(768).fill(1) }, usageMetadata: { promptTokenCount } });
    const tracker = new BudgetTracker({ label: 'google-fixture' });
    await expect(withBudgetTracker(tracker, () => embedMultimodal([{ kind: 'text', text: 'short' }]))).rejects.toThrow('shared-token');
    expect(tracker.snapshot().models[0].input_tokens).toBe(8192);
  }
});

test('Google accepts text below the local byte ceiling without changing Content bytes', async () => {
  const text = 'x'.repeat(8191);
  response = () => Response.json({ embedding: { values: Array(768).fill(1) }, usageMetadata: { promptTokenCount: 8191 } });
  expect((await embedMultimodal([{ kind: 'text', text }]))[0].length).toBe(768);
  expect(calls[0].body.content.parts).toEqual([{ text }]);
});

test('Google Safe isolates an input 400 without replaying successful Contents', async () => {
  response = () => calls[calls.length - 1].body.content.parts[0].text === 'bad'
    ? new Response('private input bytes', { status: 400 })
    : Response.json({ embedding: { values: Array(768).fill(calls.length) } });
  const tracker = new BudgetTracker({ label: 'google-fixture' });
  const result = await withBudgetTracker(tracker, () => embedMultimodalSafe(['first', 'bad', 'last'].map(text => ({ kind: 'text', text }))));
  expect(result.failedIndices).toEqual([1]);
  expect(result.embeddings.map(v => v?.[0])).toEqual([1, undefined, 3]);
  expect(calls.map(c => c.body.content.parts[0].text)).toEqual(['first', 'bad', 'last']);
  expect(tracker.snapshot().callsRecorded).toBe(3);
});

for (const status of [401, 402, 403, 404]) test(`Google Safe stops remaining Contents after HTTP ${status}`, async () => {
  response = () => calls.length === 1
    ? Response.json({ embedding: { values: Array(768).fill(1) } })
    : new Response('private input bytes', { status });
  const tracker = new BudgetTracker({ label: 'google-fixture' });
  const result = await withBudgetTracker(tracker, () => embedMultimodalSafe(
    Array.from({ length: 20 }, (_, i) => ({ kind: 'text', text: `item ${i}` })),
  ));
  expect(calls).toHaveLength(2);
  expect(result.failedIndices).toEqual(Array.from({ length: 19 }, (_, i) => i + 1));
  expect(result.embeddings.map(v => v?.[0])).toEqual([1, ...Array(19).fill(undefined)]);
  expect((result.lastError as AIConfigError).status).toBe(status);
  expect(String(result.lastError)).not.toContain('private input bytes');
  expect(tracker.snapshot().callsRecorded).toBe(2);
  expect(tracker.snapshot().models[0].input_tokens).toBe(16384);
});

for (const status of [400, 413, 422]) test(`Google Safe continues after input-specific HTTP ${status}`, async () => {
  response = () => calls.length === 2 ? new Response('private input bytes', { status })
    : Response.json({ embedding: { values: Array(768).fill(calls.length) } });
  const result = await embedMultimodalSafe(['first', 'bad', 'last'].map(text => ({ kind: 'text', text })));
  expect(result.failedIndices).toEqual([1]);
  expect(result.embeddings.map(v => v?.[0])).toEqual([1, undefined, 3]);
  expect(calls).toHaveLength(3);
});

test('Google Safe continues after a local byte refusal without billing the refused Content', async () => {
  const tracker = new BudgetTracker({ label: 'google-fixture' });
  const result = await withBudgetTracker(tracker, () => embedMultimodalSafe(
    ['first', 'x'.repeat(8192), 'last'].map(text => ({ kind: 'text', text })),
  ));
  expect(result.failedIndices).toEqual([1]);
  expect(result.embeddings.map(v => v?.[0])).toEqual([1, undefined, 2]);
  expect(calls.map(c => c.body.content.parts[0].text)).toEqual(['first', 'last']);
  expect(tracker.snapshot().callsRecorded).toBe(2);
});

test('Google rejects empty text parts even beside nonempty text before sending any Content', async () => {
  for (const empty of ['', ' \n\t']) {
    const parts = [{ kind: 'text' as const, text: 'x' }, ...Array.from({ length: 20000 }, () => ({ kind: 'text' as const, text: empty }))];
    await expect(embedMultimodal([{ kind: 'text', text: 'first' }, { kind: 'content', parts }])).rejects.toThrow('text part');
  }
  expect(calls).toEqual([]);
});

test('Google preserves nonempty text parts and their whitespace without an invented part cap', async () => {
  const parts = Array.from({ length: 64 }, (_, i) => ({ kind: 'text' as const, text: ` ${i} ` }));
  const [vec] = await embedMultimodal([{ kind: 'content', parts }]);
  expect(vec.length).toBe(768);
  expect(calls[0].body.content.parts).toEqual(parts.map(({ text }) => ({ text })));
});

// A status-only halt misses provider-wide preconditions and exhausted quotas.
// Structured daily quotas must stop immediately; a generic RESOURCE_EXHAUSTED
// or a mismatched HTTP/reason pair must not take that branch. Fetch is doubled
// at the transport boundary; no new production seam or live provider is needed.
for (const [status, error] of [
  [400, { status: 'FAILED_PRECONDITION' }],
  [400, { code: 'failed_precondition' }],
  [429, { code: 'quota_exceeded' }],
  [429, { status: 'RESOURCE_EXHAUSTED', details: [{ '@type': 'type.googleapis.com/google.rpc.ErrorInfo', reason: 'QUOTA_EXCEEDED' }] }],
  [429, { code: 429, status: 'RESOURCE_EXHAUSTED', details: [{ '@type': 'type.googleapis.com/google.rpc.QuotaFailure',
    violations: [{ quotaMetric: 'generativelanguage.googleapis.com/embed_content_free_tier_requests',
      quotaId: 'EmbedContentRequestsPerDayPerProjectPerModel-FreeTier' }] }] }],
] as const) test(`Google Safe halts on an input-independent reason ${JSON.stringify(error)}`, async () => {
  response = () => calls.length === 1
    ? Response.json({ embedding: { values: Array(768).fill(1) } })
    : Response.json({ error: { ...error, message: 'private provider echo' }, echo: 'private provider echo' }, { status });
  const tracker = new BudgetTracker({ label: 'google-fixture' });
  const result = await withBudgetTracker(tracker, () => embedMultimodalSafe(
    Array.from({ length: 20 }, (_, i) => ({ kind: 'text', text: `item ${i}` })),
  ));
  expect(calls).toHaveLength(2);
  expect(result.failedIndices).toEqual(Array.from({ length: 19 }, (_, i) => i + 1));
  expect(result.embeddings.map(v => v?.[0])).toEqual([1, ...Array(19).fill(undefined)]);
  expect(String(result.lastError) + JSON.stringify(result.lastError) + result.lastError?.stack).not.toContain('private provider echo');
  expect(tracker.snapshot().callsRecorded).toBe(2);
});

for (const [status, error] of [
  [400, { status: 'QUOTA_EXCEEDED' }],
  [429, { status: 'FAILED_PRECONDITION' }],
  [400, { code: 'quota_exceeded' }],
  [429, { details: [{ '@type': 'type.googleapis.com/google.rpc.ErrorInfo', reason: 'FAILED_PRECONDITION' }] }],
  [400, { status: 'RESOURCE_EXHAUSTED', details: [{ '@type': 'type.googleapis.com/google.rpc.QuotaFailure',
    violations: [{ quotaId: 'EmbedContentRequestsPerDayPerProjectPerModel-FreeTier' }] }] }],
  [429, { status: 'INVALID_ARGUMENT', details: [{ '@type': 'type.googleapis.com/google.rpc.QuotaFailure',
    violations: [{ quotaId: 'EmbedContentRequestsPerDayPerProjectPerModel-FreeTier' }] }] }],
  [429, { status: 'RESOURCE_EXHAUSTED', details: [{ '@type': 'type.googleapis.com/google.rpc.QuotaFailure',
    violations: [{ quotaId: 'EmbedContentRequestsPerMinutePerProjectPerModel-FreeTier' }] }] }],
  [429, { status: 'RESOURCE_EXHAUSTED', details: [{ '@type': 'unrecognized',
    violations: [{ quotaId: 'EmbedContentRequestsPerDayPerProjectPerModel-FreeTier' }] }] }],
  [429, { status: 'RESOURCE_EXHAUSTED', details: [{ '@type': 'type.googleapis.com/google.rpc.QuotaFailure',
    violations: [null, { quotaId: 42 }] }] }],
] as const) test(`Google Safe isolates mismatched HTTP/reason pairs ${status} ${JSON.stringify(error)}`, async () => {
  response = () => calls.length === 2 ? Response.json({ error }, { status })
    : Response.json({ embedding: { values: Array(768).fill(calls.length) } });
  const result = await embedMultimodalSafe(['first', 'bad', 'last'].map(text => ({ kind: 'text', text })));
  expect(calls).toHaveLength(3);
  expect(result.failedIndices).toEqual([1]);
  expect(result.embeddings.map(v => v?.[0])).toEqual([1, undefined, 3]);
});

for (const interrupt of ['input-error', 'local-refusal'] as const) {
  test(`Google Safe resets the 429 streak after ${interrupt}`, async () => {
    response = () => interrupt === 'input-error' && calls.length === 3
      ? Response.json({ error: { status: 'INVALID_ARGUMENT' } }, { status: 400 })
      : Response.json({ error: { status: 'RESOURCE_EXHAUSTED' } }, { status: 429 });
    const inputs = Array.from({ length: 8 }, (_, i) => ({ kind: 'text' as const,
      text: interrupt === 'local-refusal' && i === 2 ? '' : `item ${i}` }));
    const result = await embedMultimodalSafe(inputs);
    expect(calls).toHaveLength(interrupt === 'input-error' ? 6 : 5);
    expect(result.failedIndices).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
  });
}

for (const error of [{ status: 'INVALID_ARGUMENT' }, { code: 'invalid_request' },
  { status: 'unknown private provider echo', message: 'failed_precondition quota_exceeded' }]) {
  test(`Google Safe keeps input-specific or unknown reasons isolated ${JSON.stringify(error)}`, async () => {
    response = () => calls.length === 2 ? Response.json({ error }, { status: 400 })
      : Response.json({ embedding: { values: Array(768).fill(calls.length) } });
    const result = await embedMultimodalSafe(['first', 'bad', 'last'].map(text => ({ kind: 'text', text })));
    expect(calls).toHaveLength(3);
    expect(result.failedIndices).toEqual([1]);
    expect(result.embeddings.map(v => v?.[0])).toEqual([1, undefined, 3]);
    expect(String(result.lastError) + JSON.stringify(result.lastError)).not.toContain('private provider echo');
  });
}

test('Google Safe halts a sustained 429 streak and resets it after success', async () => {
  response = () => calls.length === 3 ? Response.json({ embedding: { values: Array(768).fill(3) } })
    : Response.json({ error: { status: 'RESOURCE_EXHAUSTED', code: 'rate_limit_exceeded' } }, { status: 429 });
  const result = await embedMultimodalSafe(Array.from({ length: 20 }, (_, i) => ({ kind: 'text', text: `item ${i}` })));
  expect(calls).toHaveLength(6);
  expect(result.embeddings.map(v => v?.[0])).toEqual([undefined, undefined, 3, ...Array(17).fill(undefined)]);
  expect(result.failedIndices).toEqual([0, 1, ...Array.from({ length: 17 }, (_, i) => i + 3)]);
});
