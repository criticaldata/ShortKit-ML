# Reproducible environment for ShortKIT-ML paper benchmarks
# Build:  docker build -t shortcut-detect .
# Run:    docker run --rm -v $(pwd)/output:/app/output shortcut-detect smoke
FROM python:3.10-slim

LABEL maintainer="ShortKIT-ML Team"
LABEL description="Reproducible environment for ShortKIT-ML paper benchmarks"

# System dependencies required by weasyprint / PDF export and general build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpango1.0-dev \
        libgdk-pixbuf2.0-dev \
        libffi-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifest first for layer caching
COPY pyproject.toml ./

# Install the package in editable mode (with all extras for reproducibility)
COPY . .
RUN pip install --no-cache-dir -e ".[dev]"

# Make the reproduce script executable
RUN chmod +x scripts/reproduce_paper.sh

# Default entrypoint runs the reproducibility script.
# Pass a profile as the CMD argument: smoke | default | full
ENTRYPOINT ["scripts/reproduce_paper.sh"]
CMD ["default"]
