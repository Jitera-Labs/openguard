FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy project metadata
COPY pyproject.toml README.md ./

# Install dependencies (using system python in the container)
# We install with [dev] because this image is also used for local dev via docker-compose
RUN uv pip install --system --no-cache -r pyproject.toml --extra dev

# Copy application code
COPY src ./src
COPY guards.yaml.example ./guards.yaml

# Install the project itself
RUN uv pip install --system --no-cache --no-deps -e .

EXPOSE 8000

# Production command: no reload
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
