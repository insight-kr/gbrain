import { afterEach, beforeEach, expect, test } from 'bun:test';
import { configureGateway, embedMultimodal, resetGateway } from '../../src/core/ai/gateway.ts';
import { __setTestRecipesForTests } from '../../src/core/ai/recipes/index.ts';
import { google } from '../../src/core/ai/recipes/google.ts';
import { AIConfigError, AITransientError } from '../../src/core/ai/errors.ts';

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
