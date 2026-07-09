# syntax=docker/dockerfile:1

# Frontend build stage
# Build frontend assets on the native build platform to avoid
# cross-architecture emulation issues during multi-platform builds.
FROM --platform=$BUILDPLATFORM oven/bun:1@sha256:e10577f0db68676a7024391c6e5cb4b879ebd17188ab750cf10024a6d700e5c4 AS frontend-builder

WORKDIR /app

# Copy frontend source code
COPY lightrag_webui/ ./lightrag_webui/

# Build frontend assets for inclusion in the API package
RUN --mount=type=cache,target=/root/.bun/install/cache \
    cd lightrag_webui \
    && bun install --frozen-lockfile \
    && bun run build

# Python build stage - using uv for faster package installation
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim@sha256:e5b65587bce7de595f299855d7385fe7fca39b8a74baa261ba1b7147afa78e58 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_SYSTEM_PYTHON=1
ENV UV_COMPILE_BYTECODE=1

WORKDIR /app

# Install system build deps. Rust was removed on purpose; every locked
# dependency in uv.lock ships manylinux wheels for x86_64 and aarch64,
# so nothing in this build compiles Rust.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:${PATH}"

# Ensure shared data directory exists for uv caches
RUN mkdir -p /root/.local/share/uv

# Copy project metadata and sources
COPY pyproject.toml .
COPY setup.py .
COPY uv.lock .

# Install base, API, and offline extras without the project to improve caching
RUN --mount=type=cache,target=/root/.local/share/uv \
    uv sync --frozen --no-dev --extra api --extra offline --no-install-project --no-editable

# Copy project sources after dependency layer
COPY lightrag/ ./lightrag/

# Include pre-built frontend assets from the previous stage
COPY --from=frontend-builder /app/lightrag/api/webui ./lightrag/api/webui

# Sync project in non-editable mode
RUN --mount=type=cache,target=/root/.local/share/uv \
    uv sync --frozen --no-dev --extra api --extra offline --no-editable

# Prepare offline cache directory and pre-populate tiktoken data
# Use uv run to execute commands from the virtual environment
RUN mkdir -p /app/data/tiktoken \
    && uv run lightrag-download-cache --cache-dir /app/data/tiktoken || status=$?; \
    if [ -n "${status:-}" ] && [ "$status" -ne 0 ] && [ "$status" -ne 2 ]; then exit "$status"; fi

# Final stage
FROM python:3.12-slim@sha256:423ed6ab25b1921a477529254bfeeabf5855151dc2c3141699a1bfc852199fbf

WORKDIR /app

# Install uv for package management
COPY --from=ghcr.io/astral-sh/uv:0.11.7@sha256:240fb85ab0f263ef12f492d8476aa3a2e4e1e333f7d67fbdd923d00a506a516a /uv /usr/local/bin/uv

ENV UV_SYSTEM_PYTHON=1

# Copy installed packages and application code
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/lightrag ./lightrag
COPY pyproject.toml .
COPY setup.py .
COPY uv.lock .

# Ensure the installed scripts are on PATH
ENV PATH=/app/.venv/bin:/root/.local/bin:$PATH

# Create persistent data directories
RUN mkdir -p /app/data/rag_storage /app/data/inputs /app/data/prompts /app/data/tiktoken

# Copy offline cache into the newly created directory
COPY --from=builder /app/data/tiktoken /app/data/tiktoken

# Point to the prepared cache
ENV TIKTOKEN_CACHE_DIR=/app/data/tiktoken
ENV WORKING_DIR=/app/data/rag_storage
ENV INPUT_DIR=/app/data/inputs
ENV PROMPT_DIR=/app/data/prompts

# Expose API port
EXPOSE 9621

# Container healthcheck against the API /health endpoint (image has no curl)
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:9621/health')" || exit 1

ENTRYPOINT ["python", "-m", "lightrag.api.lightrag_server"]
