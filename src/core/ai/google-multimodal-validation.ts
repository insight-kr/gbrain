import { AIConfigError } from './errors.ts';
import type { MultimodalPart } from './types.ts';

/** Details are keyed by modality: Google does not guarantee their array order. */
export function googleMultimodalTokenCount(metadata: unknown): number | undefined {
  if (metadata === undefined) return undefined;
  const usage = metadata as { promptTokenCount?: unknown; promptTokenDetails?: unknown };
  const invalid = () => new AIConfigError('Google shared-token usage is invalid or reaches 8192; refusing a potentially truncated embedding.');
  const valid = (n: unknown): n is number => Number.isSafeInteger(n) && (n as number) >= 0 && (n as number) < 8192;
  if (!usage || typeof usage !== 'object') throw invalid();
  const total = usage.promptTokenCount;
  if (total !== undefined && !valid(total)) throw invalid();
  const details = usage.promptTokenDetails;
  if (details === undefined) return total as number | undefined;
  if (!Array.isArray(details)) throw invalid();
  const byModality = { TEXT: 0, IMAGE: 0 };
  for (const detail of details) {
    if (!detail || !valid(detail.tokenCount)) throw invalid();
    switch (detail.modality) {
      case 'TEXT': byModality.TEXT += detail.tokenCount; break;
      case 'IMAGE': byModality.IMAGE += detail.tokenCount; break;
      default: throw invalid();
    }
  }
  if (details.length === 0) return total as number | undefined;
  const sum = byModality.TEXT + byModality.IMAGE;
  if (!valid(sum) || (total !== undefined && total !== sum)) throw invalid();
  return total as number | undefined ?? sum;
}

/** Local admission is separate from the pessimistic 8192-token spend reservation. */
export function assertGoogleMultimodalBudget(parts: readonly MultimodalPart[]): void {
  const media = parts.filter(part => part.kind === 'inline_data');
  if (media.filter(part => part.mime.startsWith('image/')).length > 6) throw new AIConfigError('Google multimodal permits at most six images per Content.');
  for (const part of media) {
    switch (part.mime) {
      case 'image/png': case 'image/jpeg':
        break;
      case 'application/pdf': case 'audio/mpeg': case 'audio/wav': case 'video/mp4': case 'video/quicktime':
        // sinaxis: media receipts are unavailable; enable after page/duration/codec and shared-token verification.
        throw new AIConfigError('Google PDF, audio and video are held until verified media and shared-token validation is available.');
      default: throw new AIConfigError('Google multimodal does not support this MIME type.');
    }
  }
  // 258 was measured on solid PNG/JPEG squares at 1–1024px, not a general
  // image upper bound. UTF-8 bytes conservatively estimate text; the provider
  // must reject any underestimate with autoTruncate=false below.
  const textBytes = parts.reduce((total, part) => total + (part.kind === 'text' ? Buffer.byteLength(part.text, 'utf8') : 0), 0);
  const estimatedTokens = textBytes + media.length * 258;
  if (estimatedTokens >= 8192) throw new AIConfigError('Google Content exceeds the local shared-token budget (258 per image plus UTF-8 text bytes, strictly below 8192).');
}
