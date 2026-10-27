# Google Gemini

GBrain calls Gemini natively for chat, query expansion and embeddings: model
strings `google:<model>`, such as `google:gemini-2.5-flash` for chat and
`google:gemini-embedding-2` for embeddings. Embedding models, dimensions and
pricing are in [embedding providers](../integrations/embedding-providers.md#google-gemini).

## Key setup

Set `GOOGLE_GENERATIVE_AI_API_KEY`, an AI Studio API key. `GEMINI_API_KEY` works as an alias
when the first variable is unset. You can also set `google_api_key` in `~/.gbrain/config.json`; a
real value in the environment wins over the config file.

## Base URL override

`GOOGLE_GENERATIVE_AI_BASE_URL` sends every native Gemini request to another
endpoint: chat, query expansion and embeddings. Use it for a metering or
logging proxy, such as an eval harness's spend proxy, or for a regional gateway.

```bash
export GOOGLE_GENERATIVE_AI_BASE_URL=http://127.0.0.1:8787
```

- **A URL with no version segment** gets `/v1beta` appended, matching the
  default `https://generativelanguage.googleapis.com/v1beta`. Requests then go
  to `<base>/v1beta/models/<model>:generateContent` (chat and expansion) and
  `:embedContent` / `:batchEmbedContents` (embeddings). A proxy that forwards
  the path upstream works unchanged.
- **A URL that already ends in a version** (`/v1`, `/v1beta`, `/v1alpha`) is
  used as written. Trailing slashes are trimmed.
- **The API key** still goes in the `x-goog-api-key` header, so the endpoint
  receives your key. Point it only at a proxy you control.
- **Unset or blank:** the SDK default is used, so nothing changes.

The file-plane equivalent is `provider_base_urls.google` in
`~/.gbrain/config.json`. The environment variable wins over it. A value merged
from a mounted brain's database never sets this URL, so a shared brain cannot
redirect your key; the same rule covers `ANTHROPIC_BASE_URL` and
`OPENAI_BASE_URL`.

Verify the override with one short request through the proxy, for example
`gbrain think "ping" --model google:gemini-2.5-flash`, and check that the proxy
logged a `/v1beta/models/gemini-2.5-flash:generateContent` call.

## Native Content preflight

`max_input_tokens['gemini-embedding-2'] = 8192` describes the shared token
limit and caps chunking for SDK text callers (the default remains 2000 tokens).
The native REST Content adapter has a separate local admission policy: **258 per
PNG/JPEG image plus the UTF-8 byte count of all text must be strictly below
8192**. At most six images are allowed. Each text part must be nonempty and
non-whitespace; original text bytes and part order are preserved. Bodies over
16 MiB and file URIs are rejected locally. PDF, audio, video and every other
image/MIME format remain held. The production recipe capability is still gated.

258 was measured on solid-color PNG/JPEG squares, 1–1024 pixels per side,
including six-image cases (1548 tokens). It is a measured budgeting assumption,
**not a general image upper bound**. UTF-8 text bytes are a conservative local
estimate, not a proof of Gemini tokenization. The adapter always sends
`embedContentConfig.autoTruncate=false`: any underestimated input must be refused
by the provider rather than silently truncated. An input-specific HTTP 400 is
isolated by the Safe batch path and does not discard earlier successful vectors.
Provider-wide failures retain their separate halt behavior.

The adapter reads `promptTokenDetails` by `modality` (`TEXT` / `IMAGE`), never by
array position. Supplied detail counts must be valid and sum to the total; a
missing total can be derived from those details. Invalid counts or a total at
or above 8192 refuse the vector. Missing usage stays conservatively estimated
for spend accounting. Usage alone does not establish an absence of truncation.
Output dimensions are locally restricted to integers 128–3072, even though the
measured endpoint accepted 127.

The isolated [measurement runner](../../scripts/measurements/gemini-embedding-2.py)
prints an offline case plan by default. It is separate from the gateway and no
default command or test invokes live measurement. Execution requires an owner
approval covering transfer decisions, endpoint, cases and a request cap.

**Say to your agent:** *"Check my image and caption against the local Gemini
budget, keeping PDF, audio and video closed."*
