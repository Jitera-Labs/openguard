FROM ghcr.io/av/tools

WORKDIR /app

# Ensure uv is available
RUN python -m pip install --no-cache-dir -U uv

COPY pyproject.toml README.md ./
COPY src ./src
COPY guards.yaml.example ./guards.yaml

RUN uv pip install --system -e .

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "/app"]
