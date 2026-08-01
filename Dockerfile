FROM python:3.12-slim

ARG UV_VERSION=0.11.16
RUN pip install --no-cache-dir "uv==${UV_VERSION}"
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY data ./data
COPY examples ./examples
RUN uv sync --frozen --no-dev \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /shared /mlflow \
    && chown -R appuser:appuser /app /shared /mlflow
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1
USER appuser
EXPOSE 8000 5000
CMD ["uvicorn", "repro_ml_pipeline.serve:app", "--host", "0.0.0.0", "--port", "8000"]
