# OpenGuard Features

**A transparent security proxy for LLM apps. Change one URL. Enforce any policy.**
**Zero code changes. Real-time guardrails. Works with every major AI provider.**

---

## Why OpenGuard

- **You're shipping AI features but have no visibility into what's going in or out.** OpenGuard intercepts every request and response before it leaves your stack.
- **Adding security means forking your LLM client or wrapping every call.** OpenGuard is a drop-in proxy — point your base URL at it and you're done.
- **PII leaks through prompts and model responses.** OpenGuard redacts sensitive data in streamed output in real time, not after the fact.
- **LLM costs spiral when users pass uncapped token limits.** `max_tokens` guard enforces a hard ceiling — adds it if absent, clamps it if too high.
- **Prompt injections and jailbreaks are hard to catch with regex alone.** `llm_input_inspection` uses a secondary LLM call to detect them structurally.

---

## Guard Types

### `content_filter`
Sanitizes blocked words out of every message, in-place.

- Replaces matched content with `[FILTERED]` across all conversation messages
- Non-blocking — request continues after sanitization
- Fast, zero latency overhead

### `pii_filter`
Strips email, phone, SSN, and credit card data before it reaches the model — and after.

- Regex-based PII detection on request messages
- Real-time redaction of streamed LLM responses via sliding window buffer
- Catches PII that spans SSE chunk boundaries

### `max_tokens`
Hard cap on the `max_tokens` parameter. Cost control with one line of YAML.

- Injects `max_tokens` if the request omits it
- Clamps to your defined ceiling if the client requests more
- Blocks runaway agent loops and prompt-stuffing abuse

### `keyword_filter`
Match plain strings or regex patterns. Block, redact, or log.

- Three modes: `block` (reject), `sanitize` (redact), `log` (audit trail)
- Detects API keys, PATs, dangerous shell commands, competitor mentions
- Patterns matched across all message roles

### `llm_input_inspection`
Routes user input to a secondary LLM with your custom policy prompt.

- Returns a structured allow/block decision with a reason
- Detects prompt injection, jailbreaks, and policy violations
- Injection-hardened — user-influenceable parameters stripped from inspector call

---

## Architecture

- FastAPI proxy on port `23294` — no framework lock-in, no SDK required.
- Guards run before forwarding to the upstream LLM; blocked requests return `400` immediately.
- Full SSE streaming proxied end-to-end — clients see no difference.
- `StreamRedactor` uses a sliding window buffer to catch patterns that span chunk boundaries.
- `ChatNode` doubly-linked list enables in-place mutation of conversation history without rebuilding it.
- Guards are dynamically loaded — drop a file in, it's available as a guard type.
- TTL-cached model discovery routes each request to the correct backend by model ID.

---

## Integration

### Deployment

| Method | Command |
|---|---|
| Zero-install | `uvx openguard` |
| Docker | `docker run ghcr.io/Jitera-Labs/openguard:main` |
| PyPI | `uv add openguard` |
| Teams | `docker-compose up` |

### Native Tool Integrations

- `openguard launch claude` — wraps Claude Code, injects `ANTHROPIC_BASE_URL`, manages proxy lifecycle automatically.
- `openguard launch opencode` — patches OpenCode config, wraps the session.
- `openguard launch codex` — wraps OpenAI Codex CLI.

### API Compatibility

- **OpenAI** `/v1/chat/completions` — drop-in replacement.
- **Anthropic** `/v1/messages` — native format, full multimodal and tool use support.
- **OpenAI Responses API** `/v1/responses` — translation layer to Chat Completions or passthrough for native providers.
- Works with OpenRouter, Ollama, local models, and any OpenAI-compatible API.

### Multi-Backend Routing

- Register unlimited backends via `OPENGUARD_OPENAI_KEY_N` / `OPENGUARD_OPENAI_URL_N` env wildcards.
- Each request is routed to the right backend by model ID — transparent to the client.

---

## Match DSL

Target guards at exactly the traffic you care about — nothing more.

Match on: model name, provider (`openai` / `anthropic`), user ID, message content, `max_tokens`, temperature, raw payload fields.

Operators: `_eq`, `_neq`, `_in`, `_ilike`, `_iregex`, `_some`, `_and`, `_or`, `_not`, `_contains`.

**Why it matters:** Apply PII filtering only to GPT-4o. Block shell commands only for unauthenticated users. Cap tokens only on free-tier requests. Granular policy without custom code.

---

## Streaming Security

OpenGuard guards don't stop working when the model starts streaming.

- SSE stream is proxied transparently — the client sees no API difference.
- `StreamRedactor` buffers a sliding window across chunks to catch patterns that span boundaries.
- PII in streamed responses is redacted before the client receives it.
- `keyword_filter` in `sanitize` mode works on streamed output, not just input.

---

## Use Cases

**AI/Agent Developer** — Ship LLM features with security policy enforced at the infra layer, not scattered across application code.

**Enterprise Security Team** — Audit, block, and redact sensitive data across all AI traffic without touching application deployments.

**SaaS Builder** — Apply per-tenant guardrails using `user` field matching — different policies for free vs. paid, trusted vs. untrusted.

**AppSec Team** — Detect prompt injection and jailbreak attempts with a secondary LLM inspection layer that's hardened against manipulation.

**Coding Tool User** — Wrap Claude Code, Codex, or OpenCode with a single command and apply security policy to all agent traffic automatically.

**Cost-Conscious Team** — Enforce `max_tokens` ceilings across all models and environments to prevent runaway bills from agents or abusive inputs.
