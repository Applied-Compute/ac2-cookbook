FROM python:3.12-alpine AS build

COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /bin/

WORKDIR /app

# google-crc32c ships no musl wheel, and the harness never computes a CRC.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    CRC32C_PURE_PYTHON=1

COPY pyproject.toml uv.lock ./
RUN --mount=type=secret,id=ac2_api_key,env=UV_INDEX_AC2_PASSWORD \
    UV_INDEX_AC2_USERNAME=__token__ \
    uv sync --frozen --no-install-project --no-dev --extra harness

COPY src ./src
RUN uv sync --frozen --no-dev --extra harness --no-editable

FROM python:3.12-alpine

RUN apk upgrade --no-cache && python -m pip uninstall --yes pip

COPY --from=build /app/.venv /app/.venv
COPY harness_server.py /app/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER 65532:65532

ENTRYPOINT ["python", "/app/harness_server.py", "--host", "0.0.0.0"]
