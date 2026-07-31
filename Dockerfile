FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY examples ./examples
RUN pip install --no-cache-dir -e .
ENTRYPOINT ["repro-ml"]
CMD ["verify-signature", "--pin", "examples/artifacts/run_signature.json"]
