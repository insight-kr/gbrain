// Protects ordered Content serialization, preflight refusal and query overrides.
// An image-default branch or ignored override fails; legacy-only tests cannot see Content.
// Uses public gateway calls and a fetch double; no new production test seam.
import { afterEach, beforeEach, expect, test } from 'bun:test';
import { configureGateway, embedMultimodal, embedQueryMultimodal, resetGateway } from '../../src/core/ai/gateway.ts';
import { AIConfigError } from '../../src/core/ai/errors.ts';

const originalFetch = globalThis.fetch;
let bodies: any[];
beforeEach(() => {
  bodies = [];
  configureGateway({ embedding_model: 'voyage:voyage-multimodal-3', env: { VOYAGE_API_KEY: 'synthetic' } });
  globalThis.fetch = (async (_url, init) => {
    const body = JSON.parse(init!.body as string);
    bodies.push(body);
    return Response.json({ data: body.inputs.map((_: unknown, i: number) => ({ embedding: Array(1024).fill(i + 1) })) });
  }) as typeof fetch;
});
afterEach(() => { globalThis.fetch = originalFetch; resetGateway(); });

test('single-part Content preserves legacy wire shape and input order', async () => {
  const vectors = await embedMultimodal([
    { kind: 'content', parts: [{ kind: 'text', text: 'first' }] },
    { kind: 'image_base64', mime: 'image/png', data: 'eA==' },
    { kind: 'text', text: 'last' },
  ]);
  expect(bodies[0].inputs).toEqual([
    { content: [{ type: 'text', text: 'first' }] },
    { content: [{ type: 'image_base64', image_base64: 'data:image/png;base64,eA==' }] },
    { content: [{ type: 'text', text: 'last' }] },
  ]);
  expect(vectors.map(v => v[0])).toEqual([1, 2, 3]);
});

test('unsupported Content is rejected before any legacy provider request', async () => {
  for (const parts of [
    [], [{ kind: 'text', text: '  ' }],
    [{ kind: 'inline_data', mime: 'image/png', data: 'bad base64' }],
    [{ kind: 'inline_data', mime: 'application/pdf', data: 'eA==' }],
    [{ kind: 'file_uri', mime: 'image/png', uri: 'https://example.test/private' }],
    [{ kind: 'text', text: 'caption' }, { kind: 'inline_data', mime: 'image/png', data: 'eA==' }],
  ]) {
    await expect(embedMultimodal([{ kind: 'content', parts } as any])).rejects.toBeInstanceOf(AIConfigError);
  }
  expect(bodies).toEqual([]);
});

test('query model override and prefix apply once without mutating global config', async () => {
  configureGateway({ embedding_model: 'openai:text-embedding-3-small', env: { VOYAGE_API_KEY: 'synthetic' } });
  await embedQueryMultimodal('query', { embeddingModel: 'voyage:voyage-multimodal-3', queryPrefix: 'Find: ' });
  expect(bodies[0].inputs).toEqual([{ content: [{ type: 'text', text: 'Find: query' }] }]);
  expect(bodies[0].input_type).toBe('query');
});
