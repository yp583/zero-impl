FROM python:3.13-slim

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv
RUN apt-get update \
 && apt-get install -y curl \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Copy files needed for package build
COPY README.md ./
COPY engine/ ./engine/
COPY data_sources/ ./data_sources/

# Install dependencies (non-editable to avoid write permission issues)
RUN uv sync --frozen --no-editable

# Set Python path
ENV PYTHONPATH=/app

# Run the dataset host
CMD ["uv", "run", "python", "-m", "data_sources.dev.dev_datahost"]
