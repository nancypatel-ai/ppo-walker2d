FROM python:3.11.10-slim

WORKDIR /workspace
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[dev]"

CMD ["python", "-m", "pytest"]
