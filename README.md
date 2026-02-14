![Splash image](./assets/splash.png)

A guarding proxy for AI that applies security and privacy controls to LLM requests.

## Features

- **Content Filtering**: Block requests containing specific words or patterns
- **PII Detection**: Detect and filter personally identifiable information
- **Token Limits**: Enforce maximum token limits per request
- **LLM Input Inspection**: Run a secondary policy check over user input before forwarding
- **Configurable via YAML**: Easy-to-manage guard rules with pattern matching

## Getting Started

### Run from GHCR image

Use the published container image as the default entrypoint:

```bash
cp guards.yaml.example guards.yaml

docker run --rm -p 23294:23294 \
  -v "$(pwd)/guards.yaml:/app/guards.yaml:ro" \
  -e OPENGUARD_CONFIG=/app/guards.yaml \
  -e OPENGUARD_OPENAI_URL_1=http://host.docker.internal:11434/v1 \
  -e OPENGUARD_OPENAI_KEY_1= \
  --add-host=host.docker.internal:host-gateway \
  ghcr.io/everlier/openguard:main
```

If your GHCR package is private, authenticate first:

```bash
echo "$GITHUB_TOKEN" | docker login ghcr.io -u <github-username> --password-stdin
```

Quick check:

```bash
curl http://localhost:23294/health
```

### Local Ollama

Start against a local Ollama backend via Harbor:

```bash
make dev-ollama
```

### Run via uvx

Run directly from this repo (no manual venv setup):

```bash
uvx --from . openguard
```

Run on a different port if `23294` is already in use:

```bash
OPENGUARD_PORT=23295 uvx --from . openguard
```

Run from PyPI (once published):

```bash
uvx openguard
```

Run from GitHub source (optional):

```bash
uvx --from git+https://github.com/everlier/openguard.git openguard
```

## Publish to PyPI

1. Create the `openguard` project on PyPI (once) at <https://pypi.org/manage/projects/>.
2. Configure a trusted publisher for this repo:
  - Owner: `everlier`
  - Repository: `openguard`
  - Workflow: `.github/workflows/pypi-publish.yaml`
  - Environment: `pypi`
3. Push a version tag (for example `v0.1.0`).
4. The `Release` workflow builds with `uv build` and publishes with `uv publish`.

Local dry run before release:

```bash
uv build
uv publish --dry-run
```

Manual publish (optional, if you are not using trusted publishing):

```bash
uv publish --token <pypi-token>
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
