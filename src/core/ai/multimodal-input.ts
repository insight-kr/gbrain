import { AIConfigError } from './errors.ts';
import type { MultimodalInput, MultimodalPart } from './types.ts';

/** Normalize legacy inputs without changing their provider-specific validation. */
export function multimodalParts(input: MultimodalInput): readonly MultimodalPart[] {
  switch (input.kind) {
    case 'text': return [{ kind: 'text', text: input.text }];
    case 'image_base64': return [{ kind: 'inline_data', mime: input.mime, data: input.data }];
    case 'content': {
      if (!Array.isArray(input.parts) || input.parts.length === 0) throw new AIConfigError('Multimodal Content must contain parts.');
      for (const part of input.parts) {
        switch (part.kind) {
          case 'text':
            if (typeof part.text !== 'string') throw new AIConfigError('Invalid multimodal text part.');
            break;
          case 'inline_data':
            if (typeof part.mime !== 'string' || !/^[a-z0-9.+-]+\/[a-z0-9.+-]+$/.test(part.mime)
              || typeof part.data !== 'string' || !part.data.length
              || Buffer.from(part.data, 'base64').toString('base64') !== part.data) {
              throw new AIConfigError('Invalid multimodal MIME or base64 data.');
            }
            break;
          case 'file_uri': throw new AIConfigError('Multimodal file_uri is not supported; supply inline data.');
          default: throw new AIConfigError('Unknown multimodal part kind.');
        }
      }
      if (input.parts.every(p => p.kind === 'text' && !p.text.trim())) throw new AIConfigError('Multimodal text Content must not be empty.');
      return input.parts;
    }
    default: throw new AIConfigError('Unknown multimodal input kind.');
  }
}

/** Existing proxy contracts support one text or image part per input. */
export function legacyMultimodalPart(input: MultimodalInput): MultimodalPart & { kind: 'text' | 'inline_data' } {
  const parts = multimodalParts(input);
  if (parts.length !== 1) throw new AIConfigError('This provider does not support compound multimodal Content.');
  const part = parts[0];
  switch (part.kind) {
    case 'text': return part;
    case 'inline_data':
      if (!part.mime.startsWith('image/')) throw new AIConfigError('This provider only supports text or image parts.');
      return part;
    case 'file_uri': throw new AIConfigError('Multimodal file_uri is not supported.');
  }
}
