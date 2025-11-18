# Contributing to HPC Pipeline

We welcome contributions to the HPC Pipeline project! This guide outlines our development standards and contribution process.

## 🚀 Getting Started

### Development Setup

```bash
# Clone the repository
git clone https://github.com/rnaarla8/hpc-pipeline.git
cd hpc-pipeline

# Setup development environment
make dev

# Install pre-commit hooks
pre-commit install
```

## 📋 Coding Standards

### Python Code Style

We follow **PEP 8** with the following additions:

```python
# Good: Clear function documentation
def init_distributed() -> tuple[int, int]:
    """Initialize distributed training environment.
    
    Returns:
        tuple[int, int]: (rank, world_size)
    """
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    return dist.get_rank(), dist.get_world_size()

# Good: Type hints for all functions
def calculate_efficiency(achieved_flops: float, theoretical_flops: float) -> float:
    """Calculate GPU efficiency percentage."""
    return (achieved_flops / theoretical_flops) * 100.0
```

### Error Handling

```python
# Good: Comprehensive error handling with logging
try:
    dist.init_process_group("nccl", init_method="env://")
except Exception as e:
    logger.warning(f"Failed to initialize distributed: {e}")
    return 0, 1
```

### Logging Standards

```python
import logging

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Use appropriate log levels
logger.info("Training started with 8 GPUs")
logger.warning("GPU utilization below 80%")
logger.error("Checkpoint corruption detected")
```

## 📊 Metrics and Observability Standards

### Prometheus Metrics Convention

```python
from prometheus_client import Gauge, Counter, Histogram

# Naming: component_metric_unit
gpu_utilization_percent = Gauge("gpu_utilization_percent", "GPU utilization", ["rank", "device"])
training_loss = Gauge("training_loss", "Training loss value", ["rank"])
checkpoint_save_seconds = Histogram("checkpoint_save_seconds", "Checkpoint save duration")

# Usage with proper labels
gpu_utilization_percent.labels(rank=0, device="cuda:0").set(85.5)
```

### OpenTelemetry Tracing

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

@tracer.start_as_current_span("training_step")
def training_step(model, data):
    with tracer.start_as_current_span("forward_pass"):
        output = model(data)
    
    with tracer.start_as_current_span("backward_pass"):
        loss.backward()
    
    return output
```

## 🧪 Testing Requirements

### Unit Tests

Every module must have corresponding unit tests with ≥85% coverage:

```python
# tests/test_module.py
import pytest
from unittest.mock import patch

class TestAMPTraining:
    def test_amp_scaling_factor(self):
        """Test AMP achieves >1.5x speedup."""
        # Implementation
        assert speedup_ratio > 1.5
    
    @patch('torch.distributed.init_process_group')
    def test_distributed_init_failure(self, mock_init):
        """Test graceful handling of distributed init failure."""
        mock_init.side_effect = Exception("Network error")
        rank, world_size = init_distributed()
        assert rank == 0 and world_size == 1
```

### Integration Tests

```python
@pytest.mark.integration
def test_orchestrator_2rank():
    """Test orchestrator with 2-rank setup."""
    # This test validates the orchestrator_2rank_test acceptance criteria
    pass

@pytest.mark.stress
def test_chaos_stress_16_ranks():
    """Test chaos scenarios with 16 ranks."""
    # This test validates the chaos_stress_tests_16_ranks acceptance criteria
    pass
```

### Performance Tests

```python
@pytest.mark.benchmark
def test_gpu_efficiency_75pct():
    """Validate GPU efficiency ≥75%."""
    efficiency = run_benchmark()
    assert efficiency >= 75.0
```

## 🏗️ Infrastructure Standards

### Helm Chart Requirements

```yaml
# charts/hpc-training/values.yaml
replicaCount: 1

image:
  repository: hpc-pipeline
  tag: latest
  pullPolicy: IfNotPresent

resources:
  limits:
    nvidia.com/gpu: 8
    memory: 512Gi
  requests:
    memory: 256Gi

# Must pass helm lint
```

### Terraform Module Standards

```hcl
# terraform/modules/hpc-cluster/main.tf
variable "cluster_size" {
  description = "Number of nodes in the cluster"
  type        = number
  default     = 4
}

variable "gpu_per_node" {
  description = "Number of GPUs per node"
  type        = number
  default     = 8
}

# Must pass terraform validate
```

## 📝 Documentation Standards

### Code Documentation

```python
def calculate_roofline_efficiency(flops: float, bandwidth: float, 
                                arithmetic_intensity: float) -> float:
    """Calculate roofline model efficiency.
    
    Args:
        flops: Achieved floating point operations per second
        bandwidth: Memory bandwidth in GB/s  
        arithmetic_intensity: FLOPs per byte ratio
        
    Returns:
        Efficiency percentage (0-100)
        
    Example:
        >>> efficiency = calculate_roofline_efficiency(100e12, 1000, 50)
        >>> print(f"Efficiency: {efficiency:.1f}%")
        Efficiency: 85.2%
    """
```

### API Documentation

Use OpenAPI 3.0 for REST APIs:

```yaml
# docs/api.yaml
openapi: 3.0.0
info:
  title: HPC Pipeline API
  version: 1.0.0
paths:
  /metrics:
    get:
      summary: Get training metrics
      responses:
        '200':
          description: Current metrics
```

## 🔄 Development Workflow

### Branch Naming

- Feature: `feature/amp-checkpointing`
- Bug fix: `fix/nccl-timeout`  
- Documentation: `docs/api-reference`

### Commit Messages

```
feat(amp): add automatic OOM recovery with grad accumulation

- Implement adaptive gradient accumulation doubling on OOM
- Add checkpoint resume validation for identical loss curves
- Integrate Prometheus metrics for OOM events

Fixes: #123
```

### Pull Request Process

1. **Create feature branch** from `main`
2. **Implement changes** following coding standards
3. **Add comprehensive tests** (unit + integration)
4. **Update documentation** if needed
5. **Run full test suite**: `make test-all`
6. **Submit PR** with detailed description

### CI/CD Pipeline

All PRs must pass:

```yaml
# .github/workflows/ci.yml (excerpt)
- name: Run Tests
  run: |
    pytest tests/ --cov=. --cov-report=xml
    
- name: Lint Code
  run: |
    black --check .
    flake8 .
    
- name: Validate Infrastructure
  run: |
    helm lint charts/
    terraform validate terraform/
```

## 🎯 Acceptance Criteria

### Component Completion Criteria

Each component must meet specific acceptance criteria:

**Memory Optimization (AMP)**:
- ✅ `amp_1_5x_faster_fp32`: AMP achieves >1.5× speedup vs FP32
- ✅ `oom_triggers_grad_accum_doubling`: OOM automatically doubles grad accumulation
- ✅ `checkpoint_resume_identical_loss`: Resume yields identical loss curve

**Distributed Training (DDP)**:  
- ✅ `cpu_sync_validation`: 2-rank CPU sync validated
- ✅ `gpu_scaling_efficiency_80pct`: 8-rank GPU efficiency ≥80%
- ✅ `rank_restart_recovery`: Recovers after rank restart

### Performance Benchmarks

- **GPU Efficiency**: ≥75% for 4-GPU setup
- **Scaling Efficiency**: ≥80% for 8-GPU distributed training
- **Checkpoint MTTR**: ≤5 minutes recovery time
- **Training Availability**: ≥99.5% uptime

## 🚨 Security Standards

### Code Security

```python
# Good: Input validation
def load_checkpoint(path: str) -> dict:
    if not path.startswith(("/checkpoints/", "/data/")):
        raise ValueError(f"Invalid checkpoint path: {path}")
    
    # Validate file integrity
    if not validate_checkpoint_hash(path):
        raise CheckpointCorruptionError(f"Corrupted checkpoint: {path}")
```

### Container Security

```dockerfile
# Dockerfile
# Use non-root user
RUN useradd -m -u 1000 hpc && usermod -aG sudo hpc
USER hpc

# Scan for vulnerabilities in CI
RUN trivy filesystem /
```

## 🏆 Recognition

Contributors who consistently follow these standards and make significant contributions will be recognized in:

- **README.md** contributors section
- **Release notes** for major contributions  
- **Monthly community updates**

## 📞 Getting Help

- **Code Review**: Tag `@maintainers` for complex changes
- **Architecture Decisions**: Create RFC in `docs/rfcs/`
- **Questions**: Use GitHub Discussions

Thank you for contributing to HPC Pipeline! 🚀
