# OpenGuard

An OpenAI-compatible guardrail proxy that applies security and privacy controls to LLM requests.

## Features

- **Content Filtering**: Block requests containing specific words or patterns
- **PII Detection**: Detect and filter personally identifiable information
- **Token Limits**: Enforce maximum token limits per request
- **Configurable via YAML**: Easy-to-manage guard rules with pattern matching

## Getting Started

Full documentation coming soon.

## Configuration

Copy `guards.yaml.example` to `guards.yaml` and customize for your needs.

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
```

## Examples

Example configurations live in [examples/](examples/). Each file is a standalone `guards.yaml` you can copy and adapt.

- [examples/01-basic-content-filter.yaml](examples/01-basic-content-filter.yaml)
- [examples/02-pii-filter-all.yaml](examples/02-pii-filter-all.yaml)
- [examples/03-max-tokens-cap.yaml](examples/03-max-tokens-cap.yaml)
- [examples/04-combined-guards.yaml](examples/04-combined-guards.yaml)
- [examples/05-advanced-matchers.yaml](examples/05-advanced-matchers.yaml)
