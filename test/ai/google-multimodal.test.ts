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
    { kind: 'content', parts: [{ kind: 'text', text: 'caption' }, { kind: 'inline_data', mime: 'image/png', data: 'eA==' }] },
    { kind: 'text', text: 'second' },
    { kind: 'image_base64', mime: 'image/jpeg', data: 'eQ==' },
  ]);
  expect(vectors.map(v => v[0])).toEqual([1, 2, 3]);
  expect(calls.map(c => c.url)).toEqual(Array(3).fill('http://127.0.0.1:9/proxy/v1beta/models/gemini-embedding-2:embedContent'));
  expect(calls[0].body).toEqual({ content: { parts: [{ text: 'caption' }, { inline_data: { mime_type: 'image/png', data: 'eA==' } }] },
    embedContentConfig: { outputDimensionality: 768, autoTruncate: false } });
  expect(calls[1].body.content.parts).toEqual([{ text: 'second' }]);
  expect(calls[2].body.content.parts).toEqual([{ inline_data: { mime_type: 'image/jpeg', data: 'eQ==' } }]);
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
