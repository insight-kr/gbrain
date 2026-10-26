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

## Native Content preflight (capability remains held)

`max_input_tokens['gemini-embedding-2'] = 8192` describes the model's shared
**token** limit and also caps chunking for SDK text callers (the default chunk
size remains 2000 tokens); it is not a usable byte allowance for a Content adapter.
The experimental native REST Content path additionally accepts at most **8191
UTF-8 text bytes across all parts**, rejects each empty or whitespace-only text
part, and holds PNG/JPEG and other media pending shared-token evidence. Nonempty
parts keep their original whitespace and order; no unverified part-count ceiling
is declared. The byte gate is a conservative local policy, not a tokenizer proof.
The SDK text path and REST Content path have different preflight contracts.
A Discovery adapter must budget serialized Content text in UTF-8 bytes before
calling REST; it must not construct samples from the recipe's token value alone.
Google multimodal capability is not advertised by the production recipe.

The isolated [measurement runner](../../scripts/measurements/gemini-embedding-2.py)
prints an offline case plan by default. It measures synthetic text/PNG/JPEG outside
the gateway so production media holds stay enforced. Its explicit execution mode
requires a later owner decision covering region, provider retention, data use and
Files API policy, an approved endpoint, selected cases and a request cap. A merged
initial external-transfer decision alone does not authorize that mode. No default
command, test suite or production entrypoint runs it.

**Say to your agent:** *"Show the offline Gemini measurement cases; keep live
requests closed until the owner approves the remaining transfer decisions."*
