// Protects the shipped recipe: enabling media must not enable other models or MIME.
import { afterEach, beforeEach, expect, test } from 'bun:test';
import { configureGateway, embedMultimodal, resetGateway } from '../../src/core/ai/gateway.ts';
import { __setTestRecipesForTests } from '../../src/core/ai/recipes/index.ts';
import { google } from '../../src/core/ai/recipes/google.ts';
import { AIConfigError } from '../../src/core/ai/errors.ts';
const originalFetch = globalThis.fetch;
let bodies: any[];
beforeEach(() => {
  bodies = [];
  configureGateway({ embedding_model: 'google:gemini-embedding-2', env: {
    GOOGLE_GENERATIVE_AI_API_KEY: 'synthetic', GOOGLE_GENERATIVE_AI_BASE_URL: 'http://127.0.0.1:9',
  } });
  globalThis.fetch = (async (_url, init) => {
    bodies.push(JSON.parse(init!.body as string));
    return Response.json({ embedding: { values: Array(768).fill(1) } });
  }) as typeof fetch;
});
afterEach(() => { globalThis.fetch = originalFetch; __setTestRecipesForTests([]); resetGateway(); });

test('the shipped Google recipe advertises only Embedding 2 with PNG and JPEG inline media', () => {
  expect(google.touchpoints.embedding?.supports_multimodal).toBe(true);
  expect(google.touchpoints.embedding?.multimodal_models).toEqual(['gemini-embedding-2']);
  expect(google.touchpoints.embedding?.multimodal_mime_types).toEqual(['image/png', 'image/jpeg']);
});

test('the production recipe sends text, PNG, JPEG and mixed Content without a test recipe override', async () => {
  const image = (mime: string) => ({ kind: 'inline_data' as const, mime, data: 'eA==' });
  const vectors = await embedMultimodal([
    { kind: 'text', text: 'query' },
    { kind: 'content', parts: [image('image/png')] },
    { kind: 'content', parts: [image('image/jpeg')] },
    { kind: 'content', parts: [{ kind: 'text', text: 'caption' }, image('image/png'), image('image/jpeg')] },
  ]);
  expect(vectors.map(v => v.length)).toEqual([768, 768, 768, 768]);
  expect(bodies.map(b => b.content.parts.length)).toEqual([1, 1, 1, 3]);
  expect(bodies.every(b => b.embedContentConfig.autoTruncate === false)).toBe(true);
});

test('the production capability refuses other models and every unsupported MIME before HTTP', async () => {
  for (const embeddingModel of ['google:gemini-embedding-001', 'google:unlisted-embedding']) {
    await expect(embedMultimodal([{ kind: 'text', text: 'query' }], { embeddingModel })).rejects.toBeInstanceOf(AIConfigError);
  }
  for (const mime of ['application/pdf', 'audio/mpeg', 'audio/wav', 'video/mp4', 'video/quicktime', 'image/webp', 'image/gif', 'image/svg+xml', 'application/octet-stream']) {
    await expect(embedMultimodal([{ kind: 'content', parts: [{ kind: 'inline_data', mime, data: 'eA==' }] }])).rejects.toBeInstanceOf(AIConfigError);
  }
  expect(bodies).toEqual([]);
});

test('the declared MIME scope can narrow provider support before HTTP', async () => {
  __setTestRecipesForTests([{ ...google, id: 'google-png-fixture', touchpoints: { ...google.touchpoints,
    embedding: { ...google.touchpoints.embedding!, multimodal_mime_types: ['image/png'] } } }]);
  const opts = { embeddingModel: 'google-png-fixture:gemini-embedding-2' };
  expect((await embedMultimodal([{ kind: 'image_base64', mime: 'image/png', data: 'eA==' }], opts))[0].length).toBe(768);
  await expect(embedMultimodal([{ kind: 'image_base64', mime: 'image/jpeg', data: 'eA==' }], opts)).rejects.toBeInstanceOf(AIConfigError);
  expect(bodies).toHaveLength(1);
});
