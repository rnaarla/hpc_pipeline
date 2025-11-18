# Multi-stage build for production HPC pipeline
ARG CUDA_VERSION=12.1
ARG PYTORCH_VERSION=2.1.0

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive

# Core dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    wget \
    git \
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
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m -u 1000 -s /bin/bash hpc
USER hpc
WORKDIR /home/hpc

# ---------------------
# Development stage
# ---------------------
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

COPY requirements-dev.txt .
RUN python3.11 -m pip install --user --upgrade pip && \
    python3.11 -m pip install --user -r requirements-dev.txt

# ---------------------
# Production stage
# ---------------------
FROM base AS production

USER hpc
COPY requirements.txt .
RUN python3.11 -m pip install --user --upgrade pip && \
    python3.11 -m pip install --user --no-cache-dir -r requirements.txt && \
    python3.11 -m pip install --user --no-cache-dir \
        torch==${PYTORCH_VERSION} \
        torchvision \
        torchaudio \
        --index-url https://download.pytorch.org/whl/cu121

COPY --chown=hpc:hpc . /home/hpc/hpc-pipeline/
WORKDIR /home/hpc/hpc-pipeline

RUN python3.11 -m pip install --user -e .

ENV PATH="/home/hpc/.local/bin:$PATH"
ENV PYTHONPATH="/home/hpc/hpc-pipeline:$PYTHONPATH"

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3.11 -c "import hpc_pipeline; print('OK')" || exit 1

CMD ["python3.11", "-m", "hpc_pipeline.orchestrator"]

# ---------------------
# CI tools stage
# ---------------------
FROM production AS ci-tools

USER root
RUN curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash && \
    curl -fsSL https://apt.releases.hashicorp.com/gpg | gpg --dearmor -o /usr/share/keyrings/hashicorp.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/hashicorp.gpg arch=$(dpkg --print-architecture)] https://apt.releases.hashicorp.com $(lsb_release -cs) main" \
      > /etc/apt/sources.list.d/hashicorp.list && \
    curl -fsSL https://aquasecurity.github.io/trivy-repo/deb/public.key | gpg --dearmor -o /usr/share/keyrings/trivy.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/trivy.gpg] https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -cs) main" \
      > /etc/apt/sources.list.d/trivy.list && \
    apt-get update && apt-get install -y terraform trivy && \
    rm -rf /var/lib/apt/lists/*
USER hpc

# ---------------------
# Testing stage
# ---------------------
FROM ci-tools AS testing

USER root
RUN apt-get update && apt-get install -y strace && rm -rf /var/lib/apt/lists/*
USER hpc

RUN python3.11 -m pip install --user -r requirements-dev.txt

CMD ["python3.11", "-m", "pytest", "tests/", "-v"]
