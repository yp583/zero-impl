FROM python:3.13-slim

RUN pip install --no-cache-dir uv
RUN apt-get update \
 && apt-get install -y curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock* ./

COPY README.md ./
COPY engine/ ./engine/
COPY data_sources/ ./data_sources/

RUN uv sync --frozen --no-editable

ENV PYTHONPATH=/app

CMD ["uv", "run", "python", "-m", "data_sources.bert.bert_datahost"]
