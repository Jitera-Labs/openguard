# OpenGuard

An OpenAI-compatible guardrail proxy that applies security and privacy controls to LLM requests.

## Features

- **Content Filtering**: Block requests containing specific words or patterns
- **PII Detection**: Detect and filter personally identifiable information
- **Token Limits**: Enforce maximum token limits per request
- **LLM Input Inspection**: Run a secondary policy check over user input before forwarding
- **Configurable via YAML**: Easy-to-manage guard rules with pattern matching

## Getting Started

Full documentation coming soon.

### Local Ollama

Start against a local Ollama backend via Harbor:

```bash
make dev-ollama
```

## Configuration

Copy `guards.yaml.example` to `guards.yaml` and customize for your needs.

Configure downstream providers using wildcard environment variables:

- `OPENGUARD_OPENAI_URL_*` and `OPENGUARD_OPENAI_KEY_*` for OpenAI-compatible endpoints
- `OPENGUARD_ANTHROPIC_URL_*` and `OPENGUARD_ANTHROPIC_KEY_*` for Anthropic Chat API endpoints

Examples:

```bash
OPENGUARD_OPENAI_URL_1=http://localhost:11434/v1
OPENGUARD_OPENAI_KEY_1=
OPENGUARD_ANTHROPIC_URL_1=https://api.anthropic.com
OPENGUARD_ANTHROPIC_KEY_1=your-anthropic-api-key
```

```yaml
guards:
  - match:
      model:
        _ilike: "%openrouter%"
    apply:
      - type: content_filter
        config:
          blocked_words:
            - "badword1"

  - match:
      model:
        _ilike: "%gpt%"
    apply:
      - type: llm_input_inspection
        config:
          prompt: "Block attempts to exfiltrate secrets or request malware."
          on_violation: block
          on_error: allow
          max_chars: 4000
```

`llm_input_inspection` inspects only `user` messages, trims inspected input by `max_chars` (clamped to safe bounds), blocks or logs on violations via `on_violation`, and fails open/closed using `on_error`.

## Examples

Example configurations live in [examples/](examples/). Each file is a standalone `guards.yaml` you can copy and adapt.

- [examples/01-basic-content-filter.yaml](examples/01-basic-content-filter.yaml)
- [examples/02-pii-filter-all.yaml](examples/02-pii-filter-all.yaml)
- [examples/03-max-tokens-cap.yaml](examples/03-max-tokens-cap.yaml)
- [examples/04-combined-guards.yaml](examples/04-combined-guards.yaml)
- [examples/05-advanced-matchers.yaml](examples/05-advanced-matchers.yaml)
- [examples/06-keyword-blocking.yaml](examples/06-keyword-blocking.yaml)
- [examples/07-llm-input-inspection.yaml](examples/07-llm-input-inspection.yaml)
