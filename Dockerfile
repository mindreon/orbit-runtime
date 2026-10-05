FROM python:3.11-slim

# docker CLI：docker 隔离模式的沙箱容器由 worker 通过宿主的 docker socket 拉起。
COPY --from=docker:27-cli /usr/local/bin/docker /usr/local/bin/docker

RUN apt-get update \
    && apt-get install -y --no-install-recommends bubblewrap \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /uvx /bin/
WORKDIR /app

COPY pyproject.toml README.md uv.lock ./
COPY packages ./packages

# --locked fails the build when uv.lock is out of date with pyproject.toml.
RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"
# Image runs either process. Compose picks the command.
#   orbit-orch
#   orbit-worker
CMD ["orbit-orch"]
