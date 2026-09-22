FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends bubblewrap \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /uvx /bin/
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY packages ./packages

RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
# Image runs either process. Compose picks the command.
#   orbit-orch
#   orbit-worker
CMD ["orbit-orch"]
