FROM python:3.13-slim

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Copy files needed for package build
COPY README.md ./
COPY engine/ ./engine/
COPY datasets/ ./datasets/
COPY test/ ./test/

# Install dependencies (non-editable to avoid write permission issues)
RUN uv sync --frozen --no-editable

# Set Python path
ENV PYTHONPATH=/app

# Run torchrun with distributed training
CMD uv run torchrun \
    --master-addr=${MASTER_ADDR} \
    --master-port=${MASTER_PORT} \
    --nnodes=1 \
    --nproc-per-node=${WORLD_SIZE} \
    test/train_dist.py
