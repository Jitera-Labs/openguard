FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy project metadata
COPY pyproject.toml README.md ./

# Install dependencies (using system python in the container)
RUN uv pip install --system --no-cache .

# Copy application code
COPY src ./src
COPY guards.yaml.example ./guards.yaml

EXPOSE 8000

# Production command: no reload
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
