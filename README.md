# HPC Pipeline for Large Language Model Training

[![CI/CD Pipeline](https://github.com/rnaarla8/hpc-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/rnaarla8/hpc-pipeline/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Production-grade HPC pipeline for training models up to 1T parameters with FAANG/NVIDIA Professional Services standards across engineering correctness, deployment reliability, and post-production operations.

## 🏗️ Architecture

The HPC pipeline consists of modular components designed for scalability and fault tolerance:

```
┌─────────────────────────────────────────────────────────────┐
│                    Orchestrator                            │
│  Config-driven CLI for end-to-end pipeline management      │
└─────────────────┬───────────────────────────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───▼───┐    ┌────▼────┐   ┌────▼────┐
│Profile│    │Training │   │ Data    │
│       │    │ (DDP/   │   │Pipeline │
│       │    │DeepSpd) │   │         │
└───────┘    └─────────┘   └─────────┘
    │             │             │
    └─────────────┼─────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│              Observability Stack                           │
│  Prometheus + Grafana + Loki + OpenTelemetry              │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

- **Memory Optimization (AMP)**: FP16/BF16 training with automatic OOM recovery
- **Distributed Training**: DDP and DeepSpeed ZeRO-3 for multi-GPU scaling  
- **Fault Tolerance**: Sharded checkpointing with hash validation
- **Observability**: Full-stack telemetry with GPU metrics and NCCL monitoring
- **Benchmarking**: Roofline analysis and Kaplan scaling law fitting

## 🚀 Quick Start (30-minute onboarding)

### Prerequisites

- CUDA 12.1+ compatible GPU
- Python 3.11+
- Docker & Docker Compose
- Kubernetes (optional, for production)

### 1. Clone and Setup

```bash
git clone https://github.com/rnaarla8/hpc-pipeline.git
cd hpc-pipeline

# Setup development environment
make dev
```

### 2. Run Basic Training

```bash
# Single GPU training with AMP
python -m optimization.amp_training --config configs/basic_training.yaml

# Multi-GPU distributed training
torchrun --nproc_per_node=2 -m optimization.amp_training --config configs/ddp_training.yaml
```

### 3. Monitor with Observability Stack

```bash
# Deploy monitoring stack
make deploy-monitoring

# Access Grafana dashboard
open http://localhost:3000
```

## 📊 Deployment Guide

### Local Development (CPU/2-GPU)

```bash
# Setup local cluster
make dev-cluster

# Run integration tests
make test-integration
```

### HPC Cluster Deployment

```bash
# Deploy with Helm
helm install hpc-training ./charts/hpc-training \
  --set image.tag=latest \
  --set resources.gpu.count=8

# Deploy observability stack
helm install monitoring ./charts/monitoring
```

### Cloud Deployment (AWS/GCP/Azure)

```bash
# Provision infrastructure with Terraform
cd terraform/
terraform init
terraform plan -var="cluster_size=16"
terraform apply

# Deploy pipeline
kubectl apply -f k8s/
```

## 🔍 Observability

### Metrics Dashboard

Access real-time training metrics:
- **GPU Utilization**: DCGM telemetry with ECC error monitoring
- **Training Progress**: Loss curves, throughput (tokens/sec), gradient norms  
- **Communication**: NCCL imbalance detection (>20% triggers alerts)
- **System Health**: Memory usage, temperature, power consumption

### Distributed Tracing

OpenTelemetry traces cover the complete pipeline:
```
Training Step → Data Loading → Forward Pass → Backward Pass → 
All-Reduce → Checkpoint Save → Metrics Export
```

### Log Aggregation

Structured logs with Loki integration:
```bash
# Query training logs
kubectl logs -l app=hpc-training | grep "ERROR\|WARN"

# View chaos events
kubectl logs -l app=chaos-monkey | grep "CHAOS"
```

## 🧪 Chaos Engineering

Validate pipeline resilience with built-in chaos testing:

```bash
# Random rank failures
python -m chaos.chaos_monkey --mode=rank_kill --interval=300s

# GPU OOM simulation  
python -m chaos.chaos_monkey --mode=gpu_oom --target_ranks=2,4

# Network slowdown injection
python -m chaos.chaos_monkey --mode=network_slow --bandwidth_limit=100mbps
```

All chaos events are logged to Prometheus with `CHAOS` tags for correlation with recovery metrics.

## 🧪 Testing

### Unit Tests
```bash
# Run all unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

### Integration Tests
```bash
# 2-rank distributed test
make test-distributed

# 16-rank stress test  
make test-stress
```

### Performance Validation
```bash
# Roofline analysis
python -m benchmarking.roofline_analysis --model_size=1B

# Scaling efficiency test
python -m benchmarking.scaling_test --min_gpus=1 --max_gpus=8
```

## 📈 Performance Benchmarks

| Model Size | GPUs | Throughput (tokens/sec) | GPU Efficiency | Memory Usage |
|------------|------|-------------------------|----------------|--------------|
| 1B params  | 2    | 15,000                 | 78%            | 14GB/GPU     |
| 7B params  | 8    | 12,000                 | 82%            | 22GB/GPU     |
| 70B params | 32   | 8,500                  | 85%            | 78GB/GPU     |

*Benchmarks on A100 80GB with NVLink interconnect*

## 🔧 Configuration

### Training Configuration
```yaml
# configs/training.yaml
model:
  size: "7B"
  architecture: "llama"

training:
  batch_size: 512
  learning_rate: 1e-4
  amp_enabled: true
  gradient_checkpointing: true

distributed:
  backend: "nccl"
  find_unused_parameters: false
  
monitoring:
  prometheus_port: 8000
  log_level: "INFO"
```

### Infrastructure Configuration
```yaml
# configs/infrastructure.yaml
cluster:
  node_count: 4
  gpu_per_node: 8
  
storage:
  checkpoint_backend: "s3"
  data_backend: "nvme"
  
networking:
  interconnect: "infiniband"
  bandwidth: "400gbps"
```

## 🚨 Troubleshooting

### Common Issues

**OOM Errors**: Pipeline automatically doubles gradient accumulation steps
```bash
# Check OOM recovery logs
kubectl logs hpc-training | grep "OOM_RECOVERY"
```

**NCCL Hangs**: Monitor communication imbalance
```bash
# View NCCL metrics in Grafana
open http://localhost:3000/d/nccl-dashboard
```

**Checkpoint Corruption**: Hash validation triggers automatic alerts
```bash
# Check checkpoint integrity
python -m fault_tolerance.checkpoint_validator --path=/checkpoints/step_1000
```

### Performance Debugging

```bash
# Profile training step
python -m profiling.pytorch_profiler --trace_steps=10

# Analyze bottlenecks
python -m benchmarking.bottleneck_analyzer --profile_data=./traces/
```

## ⚙️ GPU CI Runner

The repository contains a GitHub Actions workflow (`.github/workflows/gpu-tests.yml`) that runs the full test suite on a CUDA-enabled runner. To enable it:

1. Provision a machine (cloud or on-prem) with an NVIDIA GPU and install the matching driver plus CUDA toolkit.
2. Install Docker (optional) and the NVIDIA Container Toolkit if you plan to execute jobs inside containers.
3. Register the machine as a **self-hosted GitHub Actions runner** and tag it with `self-hosted` and `gpu`, matching the workflow’s `runs-on` requirement.
4. Pre-install project dependencies on the runner (`pip install -r requirements.txt -r requirements-dev.txt`) so subsequent jobs reuse cached wheels.
5. Push a commit or open a PR. The `gpu-tests` workflow will execute `./scripts/run_test_suite.sh` on the GPU runner and report the results in the Actions tab.

Other CI providers (GitLab, CircleCI, etc.) can reuse the same steps: attach a GPU runner, install CUDA, and execute the script.

## 🖥️ NVIDIA DGX Spark Setup

The NVIDIA DGX Spark is built on the Grace Blackwell architecture (20-core Arm CPU + Blackwell GPU, unified LPDDR5x memory). Use the steps below to bring this repository up on a DGX Spark and run the full test workflow:

1. **Base System Preparation**
   - Install the latest NVIDIA DGX OS (or another supported Linux distribution).
   - Install the NVIDIA driver and the CUDA toolkit matching the Blackwell GPU (CUDA 12.5+).
   - Verify GPU visibility:
     ```bash
     nvidia-smi
     nvcc --version
     ```

2. **Python Environment (aarch64)**
   - Install Miniconda for ARM or use NVIDIA’s Conda distribution.
   - Create and activate a Python 3.11 environment:
     ```bash
     conda create -n hpc-pipeline python=3.11
     conda activate hpc-pipeline
     ```
   - Install PyTorch with CUDA support for aarch64 (via NVIDIA wheels or compiled from source), for example:
     ```bash
     pip install --extra-index-url https://download.pytorch.org/whl/cu125 torch torchvision torchaudio
     ```
   - Install project dependencies:
     ```bash
     pip install -r requirements.txt
     pip install -r requirements-dev.txt
     ```

3. **Custom Kernel Build**
   ```bash
   python setup.py build_ext --inplace
   ```

4. **Run Full Test Suite**
   ```bash
   ./scripts/run_test_suite.sh
   ```
   This validates the CUDA toolchain and all project tests on the DGX Spark hardware.

5. **Optional: Register DGX Spark as GitHub Actions Runner**
   - Install the GitHub Actions runner (ARM build) on the DGX Spark host:
     ```bash
     mkdir actions-runner && cd actions-runner
     curl -O https://github.com/actions/runner/releases/download/v2.317.0/actions-runner-linux-arm64-2.317.0.tar.gz
     tar xzf actions-runner-linux-arm64-2.317.0.tar.gz
     ./config.sh --url https://github.com/<org>/<repo> --token <token> --labels "self-hosted,gpu,arm64"
     sudo ./svc.sh install
     sudo ./svc.sh start
     ```
   - Ensure the `hpc-pipeline` Conda environment is available to the runner service (e.g., source it in the runner service script or use a wrapper).
   - The existing `gpu-tests` workflow will now execute on the DGX Spark for every push/PR.

6. **Confirm End-to-End**
   ```bash
   # Validate CUDA access
   nvidia-smi
   python - <<'PY'
   import torch
   assert torch.cuda.is_available(), "CUDA not available"
   print("CUDA devices:", torch.cuda.device_count())
   PY

   # Run local test suite
   ./scripts/run_test_suite.sh
   ```
   - Push a commit or open a PR and verify that the `gpu-tests` workflow passes in GitHub Actions.

## 🍳 Cookbook: End-to-End Bring-Up (DGX Spark)

Follow the steps below verbatim to take a factory-fresh NVIDIA DGX Spark from bare metal to a fully tested `hpc-pipeline` deployment with GPU CI integration.

### 1. System Preparation

```bash
# Update the base operating system
sudo apt update
sudo apt -y upgrade
sudo reboot
```

After the reboot, confirm you are running the expected kernel:
```bash
uname -a
```

### 2. Install NVIDIA Drivers and CUDA Toolkit (Arm64 / SBSA)

1. Install prerequisites and add the NVIDIA CUDA repository for Ubuntu 22.04 SBSA (Arm64):
   ```bash
   sudo apt-get install -y curl gnupg ca-certificates
   sudo mkdir -p /etc/apt/keyrings
   curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/3bf863cc.pub |
     sudo gpg --yes --dearmor -o /etc/apt/keyrings/cuda-sbsa-keyring.gpg
   echo "deb [signed-by=/etc/apt/keyrings/cuda-sbsa-keyring.gpg arch=arm64] \
   https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/ /" |
     sudo tee /etc/apt/sources.list.d/cuda-sbsa.list
   sudo apt-get update
   ```

2. Install the NVIDIA driver (update the version to the latest GA if necessary):
   ```bash
   sudo apt-get install -y nvidia-driver-555
   sudo reboot
   ```

3. Install the CUDA toolkit (version 12.5 or newer for Blackwell):
   ```bash
   sudo apt-get install -y cuda-toolkit-12-5
   ```

4. Configure environment variables so CUDA binaries and libraries are on your path:
   ```bash
   echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

5. Validate the installation:
   ```bash
   nvidia-smi
   nvcc --version
   ```

### 3. Install Miniconda (Arm64) and Create the Project Environment

```bash
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -o miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
rm miniconda.sh
eval "$($HOME/miniconda/bin/conda shell.bash hook)"
conda init bash
exec "$SHELL"  # reload shell with conda support
```

Create and activate the project environment:
```bash
conda create -y -n hpc-pipeline python=3.11
conda activate hpc-pipeline
```

### 4. Install PyTorch and Project Dependencies (Arm64 + CUDA)

1. Install PyTorch built for CUDA 12.5 on Arm64 (replace with the most recent wheel when available):
   ```bash
   pip install --upgrade pip
   pip install --extra-index-url https://download.pytorch.org/whl/cu125 torch torchvision torchaudio
   ```

2. Verify that PyTorch can see the GPU:
   ```bash
   python - <<'PY'
   import torch
   assert torch.cuda.is_available(), "CUDA not visible to PyTorch"
   print("CUDA devices:", torch.cuda.device_count())
   print("Current device:", torch.cuda.get_device_name(0))
   PY
   ```

3. Clone the repository and install dependencies:
   ```bash
   git clone https://github.com/rnaarla8/hpc-pipeline.git
   cd hpc-pipeline
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. (Optional) If you rely on editable installs or custom modules, add the project root to `PYTHONPATH`:
   ```bash
   echo "export PYTHONPATH=$(pwd):\$PYTHONPATH" >> ~/.bashrc
   source ~/.bashrc
   ```

### 5. Build Custom CUDA Extensions

```bash
conda activate hpc-pipeline  # ensure the env is active
python setup.py build_ext --inplace
```

If you need to target a specific compute capability, set `TORCH_CUDA_ARCH_LIST` before the build (Blackwell uses `sm_100`):
```bash
export TORCH_CUDA_ARCH_LIST="sm_100"
python setup.py build_ext --inplace
```

### 6. Run the Full Validation Suite

```bash
./scripts/run_test_suite.sh
```

This executes every unit test, integration test, and the production readiness suite (the latter will skip automatically if certain CUDA prerequisites are unavailable).

### 7. Register DGX Spark as a GitHub Actions Runner (optional but recommended)

1. Install the ARM64 runner binary:
   ```bash
   mkdir actions-runner && cd actions-runner
   curl -O https://github.com/actions/runner/releases/download/v2.317.0/actions-runner-linux-arm64-2.317.0.tar.gz
   tar xzf actions-runner-linux-arm64-2.317.0.tar.gz
   ```

2. Configure the runner (replace `<org>` and `<repo>` with your GitHub org/repo and supply the registration token from the repository settings):
   ```bash
   ./config.sh --url https://github.com/<org>/<repo> --token <registration-token> --labels "self-hosted,gpu,arm64"
   ```

3. Install and start the runner as a system service:
   ```bash
   sudo ./svc.sh install
   sudo ./svc.sh start
   ```

4. Ensure the runner launches inside the Conda environment. The simplest approach is to prepend `source $HOME/miniconda/bin/activate hpc-pipeline` to the `runsvc.sh` script or to set `RUNNER_ENVIRONMENT` variables.

5. Warm up dependency caches so CI runs quickly:
   ```bash
   conda activate hpc-pipeline
   pip install -r requirements.txt -r requirements-dev.txt
   ```

### 8. Execute GPU CI Workflow

1. Confirm CUDA is available to the runner:
   ```bash
   nvidia-smi
   python - <<'PY'
   import torch
   assert torch.cuda.is_available(), "Runner cannot see CUDA devices"
   PY
   ```

2. Push any commit or open a pull request. GitHub Actions will pick up `.github/workflows/gpu-tests.yml` and execute `./scripts/run_test_suite.sh` on the DGX Spark runner.

3. Monitor the workflow in the Actions tab to ensure green runs before merging changes.

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines and coding standards.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/rnaarla8/hpc-pipeline/issues)
- **Discussions**: [GitHub Discussions](https://github.com/rnaarla8/hpc-pipeline/discussions)

---

// End of Selection
```
