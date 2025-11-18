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

## ⚙️ Operations & Tooling

- **GPU runner bootstrap:** use [`scripts/setup_gpu_runner.sh`](scripts/setup_gpu_runner.sh) to install drivers, CUDA, Docker, NVIDIA Container Toolkit, Helm/Terraform/Trivy, and (optionally) auto-register a self-hosted GitHub Actions runner. The full cookbook—including DGX Spark–specific steps—is documented in [`docs/setup/gpu-runner.md`](docs/setup/gpu-runner.md).
- **Local guardrail:** run `make validate` before pushing. It enforces formatting, linting, Helm/Terraform validation (when installed), and executes CPU-only unit tests via `pytest -m "not gpu"`.
- **Full suite:** execute `./scripts/run_test_suite.sh` to run unit, integration, GPU, and production-readiness suites; GitHub Actions reuses the same command on `[self-hosted, gpu]` runners.

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines and coding standards.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/rnaarla8/hpc-pipeline/issues)
- **Discussions**: [GitHub Discussions](https://github.com/rnaarla8/hpc-pipeline/discussions)

---
