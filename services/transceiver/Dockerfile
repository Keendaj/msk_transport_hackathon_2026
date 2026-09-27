FROM python:3.14-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0
WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY services/transceiver/pyproject.toml services/transceiver/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package transceiver --no-install-workspace

COPY services/transceiver services/transceiver
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package transceiver --no-editable


FROM python:3.14-slim
RUN useradd --create-home app
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1
USER app

EXPOSE 8000 9000
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
CMD ["uvicorn", "transceiver.main:app", "--host", "0.0.0.0", "--port", "8000"]
