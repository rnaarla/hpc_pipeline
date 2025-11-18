## GPU Runner Cookbook (DGX Spark Example)

This guide walks through provisioning a Grace-Blackwell DGX Spark (or any comparable NVIDIA GPU server) so it can run the full `hpc-pipeline` test suite and host GitHub Actions GPU jobs.

### 1. Prepare the Operating System

```bash
sudo apt update
sudo apt -y upgrade
sudo reboot
```

After reboot:

```bash
uname -a
```

### 2. Run the Bootstrap Script

Copy the repository to the machine and execute the GPU runner bootstrapper:

```bash
git clone https://github.com/rnaarla/hpc_pipeline.git
cd hpc_pipeline
sudo bash scripts/setup_gpu_runner.sh
```

You can override behavior with environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VERSION` | `12.5` | CUDA toolkit version to install |
| `PYTHON_VERSION` | `3.11` | Python interpreter version |
| `GH_RUNNER_URL` | unset | GitHub repository/org URL for actions runner |
| `GH_RUNNER_TOKEN` | unset | Registration token for the runner |
| `GH_RUNNER_LABELS` | `self-hosted,gpu` | Labels applied to the runner |
| `GH_RUNNER_NAME` | `<hostname>-gpu` | Runner display name |
| `GH_RUNNER_DIR` | `/opt/actions-runner` | Runner installation directory |

If `GH_RUNNER_URL` and `GH_RUNNER_TOKEN` are set, the script automatically configures and starts the GitHub Actions runner service.

### 3. Validate CUDA and Docker

```bash
nvidia-smi
nvcc --version

python3.11 - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not visible to PyTorch"
print("CUDA device:", torch.cuda.get_device_name(0))
PY
```

### 4. Clone and Build the Project

```bash
git clone https://github.com/rnaarla8/hpc-pipeline.git
cd hpc-pipeline
python3.11 -m pip install -r requirements.txt
python3.11 -m pip install -r requirements-dev.txt
python3.11 setup.py build_ext --inplace
```

### 5. Run the Full Test Suite

```bash
./scripts/run_test_suite.sh
```

### 6. (Optional) Register the Runner Manually

If you did not supply `GH_RUNNER_URL` and `GH_RUNNER_TOKEN` to the script, follow these steps:

```bash
export RUNNER_DIR=/opt/actions-runner
sudo mkdir -p "${RUNNER_DIR}"
cd "${RUNNER_DIR}"
curl -fsSL -o actions-runner.tar.gz \
  https://github.com/actions/runner/releases/download/v2.317.0/actions-runner-linux-arm64-2.317.0.tar.gz
sudo tar xzf actions-runner.tar.gz
sudo ./config.sh --url https://github.com/<org>/<repo> \
  --token <registration-token> \
  --labels "self-hosted,gpu" \
  --name "$(hostname)-gpu" \
  --unattended
sudo ./svc.sh install
sudo ./svc.sh start
```

### 7. Confirm GitHub Actions Integration

Push a commit or open a pull request in the repository. The `gpu-tests` job in `.github/workflows/ci.yml` (and the dedicated `gpu-tests.yml`) should target the new runner automatically.

---

For customized environments (multi-GPU clusters, x86-based hosts, etc.), adapt the bootstrap script with the correct architecture tarball (`actions-runner-linux-x64-…`) and CUDA repository (e.g., `cuda/repos/ubuntu2204/x86_64`).

