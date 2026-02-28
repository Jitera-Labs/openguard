FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1

COPY pyproject.toml README.md ./
RUN uv pip install --system --no-cache -r pyproject.toml

COPY src ./src
COPY guards.yaml.example ./guards.yaml
COPY presets ./presets

RUN uv pip install --system --no-cache --no-deps .

RUN addgroup --system nonroot && adduser --system --ingroup nonroot nonroot
USER nonroot

EXPOSE 23294
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "23294"]

FROM base AS dev

USER root

# In dev, the repo is bind-mounted at /app.  Ensure the live source tree
# takes precedence over the stale site-packages copy baked into the image.
ENV PYTHONPATH=/app

# Dev/test Python dependencies (pytest, ruff, mypy, livereload)
RUN uv pip install --system --no-cache -r pyproject.toml --extra dev

# Install Node.js 20, bun, git, and gh CLI
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl gnupg unzip \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list \
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
       | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
       > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends nodejs git gh \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://bun.sh/install | bash \
    && ln -s /root/.bun/bin/bun /usr/local/bin/bun

# Agentic harness: npm tools
RUN npm install -g \
    @anthropic-ai/claude-code \
    @openai/codex \
    opencode-ai \
    @google/gemini-cli

# Agentic harness: Python tools
RUN uv pip install --system --no-cache \
    aider-chat \
    mistral-vibe

# gh copilot extension can be installed at runtime: gh extension install github/gh-copilot
