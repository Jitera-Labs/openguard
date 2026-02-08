FROM ghcr.io/av/tools

WORKDIR /app

# uv is already available in the base image

COPY pyproject.toml README.md ./
COPY src ./src
COPY guards.yaml.example ./guards.yaml

RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH

RUN uv pip install -e .

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "/app"]
