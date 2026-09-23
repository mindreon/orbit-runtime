FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends bubblewrap \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /uvx /bin/
WORKDIR /app

COPY pyproject.toml README.md ./
COPY uv.lock.parts ./uv.lock.parts
COPY packages ./packages

# The root lock is assembled here so the image matches a checkout that stores
# the lock in parts (GitHub's file API rejects one 600KB payload).
RUN cat uv.lock.parts/part-* > uv.lock \
    && sha256sum -c uv.lock.parts/SHA256SUMS \
    && uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
# Image runs either process. Compose picks the command.
#   orbit-orch
#   orbit-worker
CMD ["orbit-orch"]
