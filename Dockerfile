# Multi-stage build for production HPC pipeline
ARG CUDA_VERSION=12.1
ARG PYTORCH_VERSION=2.1.0

# Base CUDA runtime image
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libopenmpi-dev \
    openmpi-bin \
    openssh-client \
    libssl-dev \
    libffi-dev \
    libnuma1 \
    numactl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash hpc && \
    usermod -aG sudo hpc
USER hpc
WORKDIR /home/hpc

# Development stage
FROM base AS development

USER root
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    tmux \
    gdb \
    valgrind \
    && rm -rf /var/lib/apt/lists/*

USER hpc

# Install Python dependencies for development
COPY requirements-dev.txt .
RUN python3.11 -m pip install --user -r requirements-dev.txt

# Production stage
FROM base AS production

USER hpc

# Copy requirements and install production dependencies
COPY requirements.txt .
RUN python3.11 -m pip install --user --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support
RUN python3.11 -m pip install --user --no-cache-dir \
    torch==${PYTORCH_VERSION} \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Copy application code
COPY --chown=hpc:hpc . /home/hpc/hpc-pipeline/
WORKDIR /home/hpc/hpc-pipeline

# Install package in editable mode
RUN python3.11 -m pip install --user -e .

# Set Python path
ENV PATH="/home/hpc/.local/bin:$PATH"
ENV PYTHONPATH="/home/hpc/hpc-pipeline:$PYTHONPATH"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3.11 -c "import hpc_pipeline; print('OK')" || exit 1

# Default command
CMD ["python3.11", "-m", "hpc_pipeline.orchestrator"]

# Testing stage
FROM production AS testing

USER root
RUN apt-get update && apt-get install -y \
    strace \
    && rm -rf /var/lib/apt/lists/*

USER hpc

# Install test dependencies
COPY requirements-dev.txt .
RUN python3.11 -m pip install --user -r requirements-dev.txt

# Copy test files
COPY --chown=hpc:hpc tests/ tests/

# Run tests by default
CMD ["python3.11", "-m", "pytest", "tests/", "-v"]
