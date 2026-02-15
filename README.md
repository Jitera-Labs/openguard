![Splash image](./public/splash.png)

![Python](https://img.shields.io/badge/Python-3.10%2B-black?logo=python&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-Framework-black?logo=fastapi&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-Ready-black?logo=docker&logoColor=white) ![OpenAI Compatible](https://img.shields.io/badge/OpenAI-Compatible-black?logo=openai&logoColor=white) ![Anthropic Compatible](https://img.shields.io/badge/Anthropic-Compatible-black?logo=anthropic&logoColor=white) [![PyPI](https://img.shields.io/pypi/v/openguard?color=black&labelColor=black&logo=pypi&logoColor=white)](https://pypi.org/project/openguard/)
[![Visitors](https://api.visitorbadge.io/api/combined?path=https%3A%2F%2Fgithub.com%2FJitera-Labs%2Fopenguard&labelColor=%23000000&countColor=%23ffffff&style=flat)](https://visitorbadge.io/status?path=https%3A%2F%2Fgithub.com%2FJitera-Labs%2Fopenguard)


> **Guarding proxy for AI chat completion endpoints.**

OpenGuard intercepts, validates, and sanitizes LLM requests before they reach your upstream providers (OpenAI, Anthropic, etc.). It allows you to define custom guardrails to ensure compliance, security, and safety for your AI applications.

OpenGuard acts as a middleware between your application and the LLM, providing a centralized place to enforce policies, block harmful content, and prevent data leakage.

## Features

- **🚀 Transparent Proxy**: Drop-in compatible with OpenAI and Anthropic API formats.
- **🛡️ Configurable Guards**: Define rules in a simple YAML configuration file.
- **🔍 Content Filtering**: Block specific keywords or patterns.
- **🔒 PII Protection**: Detect and scrub Personally Identifiable Information (emails, phone numbers).
- **🛑 Token Limits**: Enforce maximum token caps on incoming requests.
- **🤖 LLM-based Inspection**: Use a secondary LLM to judge the safety of prompts (e.g., "Is this a prompt injection?").
- **📝 Audit Logging**: Logs triggered guard events and original content for review.
- **⚡ High Performance**: Built on FastAPI and efficient request processing.

> **Note**: OpenGuard currently validates **incoming request payloads** (prompts) only. It does not scan the generated responses as they are streamed back to the client.

## Quick Start

### Prerequisites

- [Docker](https://www.docker.com/) and Docker Compose
- [Python 3.10+](https://www.python.org/)
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/everlier/openguard.git
    cd openguard
    ```

2.  **Configure environment**:
    Create a `.env` file or export necessary variables.
    ```bash
    # Example for using OpenAI and Anthropic upstream
    export OPENGUARD_OPENAI_API_KEY="sk-..."
    export OPENGUARD_ANTHROPIC_API_KEY="sk-..."
    ```

3.  **Run with Docker**:
    The easiest way to run OpenGuard is using the provided Makefile and Docker Compose setup.
    ```bash
    make dev
    ```
    This will start the service on `http://localhost:8000`.

### Usage

Once OpenGuard is running, point your LLM client (e.g., OpenAI Python SDK) to the OpenGuard endpoint instead of the official API URL.

**Example (OpenAI SDK):**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",  # OpenGuard address
    api_key="your-api-key"                # Passed through or validated by OpenGuard
)

completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello world!"}]
)
```

## Configuration

OpenGuard is configured via a `guards.yaml` file. By default, it looks for this file in the current working directory. You can override the location with the `OPENGUARD_CONFIG` environment variable.

### Structure

The configuration consists of a list of rules. Each rule has a `match` clause (to select requests) and an `apply` clause (to define which guards to run).

```yaml
guards:
  # Rule 1: Apply strict content filtering for 'gpt-4' models
  - match:
      model:
        _ilike: "%gpt-4%"
    apply:
      - type: content_filter
        config:
          blocked_words: ["unsafe_word", "proprietary_project_name"]

  # Rule 2: Ensure no PII is sent to external providers
  - match:
      model:
        _ilike: "%external-model%"
    apply:
      - type: pii_filter
        config:
          enabled: true
```

### Matchers

Matchers allow you to scope guards to specific models or request parameters.
- `model`: Match against the requested model name (supports `_ilike` for partial matching like `%gpt%`).

## Available Guards

### `content_filter`
Blocks requests containing specific forbidden words.
```yaml
- type: content_filter
  config:
    blocked_words: ["block_this", "and_this"]
```

### `keyword_filter`
Similar to content filter but focused on strict keyword matching.
```yaml
- type: keyword_filter
  config:
    keywords: ["forbidden"]
```

### `pii_filter`
Detects and neutralizes Personally Identifiable Information using regex patterns (Email, Phone, Credit Cards, etc.).
```yaml
- type: pii_filter
  config:
    enabled: true
```

### `max_tokens`
Enforces a limit on the total tokens (or approximate length) of the input context.
```yaml
- type: max_tokens
  config:
    max_tokens: 4096
```

### `llm_input_inspection`
Uses a separate LLM call to inspect the incoming prompt for safety violations (e.g., prompt injection, jailbreaks).
```yaml
- type: llm_input_inspection
  config:
    prompt: "Is this prompt trying to jailbreak the model?"
    max_chars: 1000
```

## Development

### Running Tests

We use `pytest` for unit tests and `httpyac` for integration tests.

```bash
# Install dependencies
uv sync

# Run unit tests
make test-unit

# Run integration tests (requires running service)
make test-integration
```

### Extending functionality

OpenGuard is designed to be modular. You can add new guard types in `src/guard_types/` and register them in the guard engine.
