## OpenGuard

OpenGuard is an OpenAI-compatible guardrail proxy that allows to define custom guardrails for chat completion endpoints. It'll scan the payload and apply the defined guardrails to ensure that the responses adhere to specified rules and guidelines.

OpenGuard is based on the Open Source codebase from: ./reference/boost,
as it already contains all necessary components to implement the guardrail functionality.

The backend is essentially a Boost (mentioned codebase) + a custom module for implementing specified guardrails from a configuration file.

See additional documentation about Boost in the ./reference/docs folder.

## Implementation

- Full Boost (mentioned reference) code as a starting point
- Built-in guardrail Boost module for applying guardrails to incoming payloads

### Scope

Guards apply to **incoming chat completion payloads only**, not to responses. Since responses are streamed back to clients, there's no clear point at which scanning can be performed. The entire request payload can be validated, not just messages.

## Guardrail Module

Implemented in a transducer style that applies guarding middlewares based on the set of matchers:

```yaml
guards:
  - match:
      model:
        ilike: %openrouter%
    apply:
      - type: content_filter
        config:
          blocked_words:
            - "badword1"
            - "badword2"
      - type: response_length
        config:
          max_length: 1000
  - match:
      model:
        ilike: %kimi%
    apply:
      - type: pii_filter
```

Matchers use Hasura-style filtering with nested fields and comparisons, based on the ./reference/selection.py module.

### Guard Behavior

When a request matches a guard, the guard transforms the payload using one of these approaches:

1. **Content Neutralization**: Replace problematic content with safe placeholders (e.g., `<protected email>` for PII) and inject system instructions so the LLM understands what was filtered
2. **Payload Modification**: Modify request parameters (e.g., enforce max token limits, remove specific fields)
3. **Synthetic Response**: Return a stubbed response directly to the client without calling the LLM

Original content is logged for auditing purposes using standard logging output.

### Built-in Guard Types

Initial minimal set of guards suitable for LLM privacy and security:

- **content_filter**: Block or replace specific words/patterns
- **pii_filter**: Detect and neutralize personally identifiable information (emails, phone numbers, etc.)
- **max_tokens**: Enforce maximum token limits on requests

Guards use a modular structure similar to Boost custom modules, making it easy to add new guard types. Future extensions will include guards with inference capabilities and application-specific modules.

## Configuration

Guards are configured via a YAML file with the following default behavior:
- Default path: `./guards.yaml` (relative to working directory)
- Override via environment variable: `OPENGUARD_CONFIG`

Example configuration:
```yaml
guards:
  - match:
      model:
        ilike: %openrouter%
    apply:
      - type: content_filter
        config:
          blocked_words:
            - "badword1"
            - "badword2"
      - type: max_tokens
        config:
          max_tokens: 1000
  - match:
      model:
        ilike: %kimi%
    apply:
      - type: pii_filter
```

### Audit Logging

When guards trigger, the original content is logged to stdout/stderr using the existing logging infrastructure. Future versions may support external notification systems.

## Codebase shape

- Do not mention Harbor Boost as a source of code, just base the implementation on it. I'm `av`, the original author of Harbor Boost, it's my intent to reuse that codebase in such a way.
- Easy extension with custom guard types and matchers, so the code should be modular and well-structured.
- Use `uv` for any kind of dependency management.
- Use `pytest` for testing.
- Use `httpx` for HTTP requests in tests.
- Use `pydantic` for configuration management and validation.
