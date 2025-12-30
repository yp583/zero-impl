FROM python:3.13-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock* ./

COPY README.md ./
COPY engine/ ./engine/
COPY data_sources/ ./data_sources/
COPY test/ ./test/

RUN uv sync --frozen --no-editable

ENV PYTHONPATH=/app

CMD uv run torchrun \
    --master-addr=${MASTER_ADDR} \
    --master-port=${MASTER_PORT} \
    --nnodes=1 \
    --nproc-per-node=${WORLD_SIZE} \
    test/test_mem/train_dist.py
