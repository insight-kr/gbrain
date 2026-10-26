import { AIConfigError } from './errors.ts';

/** A provider-wide precondition or quota failure cannot improve with different input. */
export class GoogleMultimodalBatchError extends AIConfigError {}

/** Retain only recognized enum values, never upstream messages or response objects. */
export async function googleMultimodalError(response: Response): Promise<Error> {
  const status = response.status;
  let reason: 'FAILED_PRECONDITION' | 'QUOTA_EXCEEDED' | undefined;
  if (status === 400 || status === 429) {
    try {
      const payload = await response.json();
      const error = payload?.error;
      const values: unknown[] = [error?.status, error?.code];
      if (Array.isArray(error?.details)) {
        for (const detail of error.details) {
          if (detail?.['@type'] === 'type.googleapis.com/google.rpc.ErrorInfo') values.push(detail.reason);
          // RESOURCE_EXHAUSTED also covers temporary rate limits. Only a
          // structured EmbedContent daily quota proves an immediate halt.
          if (status === 429 && error?.status === 'RESOURCE_EXHAUSTED'
            && detail?.['@type'] === 'type.googleapis.com/google.rpc.QuotaFailure'
            && Array.isArray(detail.violations) && detail.violations.some((violation: unknown) => {
              const quotaId = (violation as { quotaId?: unknown } | null)?.quotaId;
              return typeof quotaId === 'string' && /^EmbedContent\w*PerDay\w*(?:-\w+)?$/.test(quotaId);
            })) reason = 'QUOTA_EXCEEDED';
        }
      }
      for (const value of values) {
        if (typeof value !== 'string') continue;
        const candidate = value.toUpperCase();
        if (status === 400 && candidate === 'FAILED_PRECONDITION') reason = 'FAILED_PRECONDITION';
        if (status === 429 && candidate === 'QUOTA_EXCEEDED') reason = 'QUOTA_EXCEEDED';
      }
    } catch { /* A malformed error body falls back to HTTP status alone. */ }
  }
  if (reason) return Object.assign(new GoogleMultimodalBatchError(
    `Google multimodal returned HTTP ${status} (${reason}).`,
    'Check the provider project prerequisites, billing and quota before retrying.',
  ), { status });
  return Object.assign(new Error(`Google multimodal returned HTTP ${status}.`), { status });
}
