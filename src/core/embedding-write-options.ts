import type { BrainEngine } from './engine.ts';
import { embeddingWriteTarget } from './page-state/projections.ts';
import { currentEmbeddingSignature } from './embedding.ts';
import type { ResolvedColumn } from './types.ts';
import type { EmbedBatchOptions } from './embedding.ts';

/** Bind Discovery text writes to the same model and width as its queries. */
export function embeddingWriteOptions(column: ResolvedColumn): Pick<EmbedBatchOptions, 'embeddingModel' | 'dimensions'> {
  return column.name === 'embedding_discovery'
    ? { embeddingModel: column.embeddingModel, dimensions: column.dimensions }
    : {};
}

/** Primary text retains its gateway signature; Discovery records its actual space. */
export function embeddingWriteSignature(column: ResolvedColumn, fallback: string | null | undefined): string | null | undefined {
  return column.name === 'embedding_discovery' ? `${column.embeddingModel}:${column.dimensions}` : fallback;
}

/** Resolve the write space before cost estimates, drift probes and completion stamps. */
export async function currentEmbeddingWriteSignature(engine: Pick<BrainEngine, 'executeRaw'>): Promise<string | null> {
  return embeddingWriteSignature((await embeddingWriteTarget(engine)).column, currentEmbeddingSignature()) ?? null;
}

/** Capture provider options and the corresponding signature from one target read. */
export async function resolveEmbeddingWriteSpace(engine: Pick<BrainEngine, 'executeRaw'>, fallback?: string | null) {
  const { column } = await embeddingWriteTarget(engine);
  return { options: embeddingWriteOptions(column), signature: embeddingWriteSignature(column, fallback) ?? undefined };
}
