#!/usr/bin/env bash
set -euo pipefail

# Usage: sudo ./scripts/setup_gpu_runner.sh
# Installs NVIDIA driver, CUDA toolkit, Docker + NVIDIA Container Toolkit,
# Helm, Terraform, Trivy, Python 3.11, and the optional GitHub Actions runner
# service when GH_RUNNER_URL and GH_RUNNER_TOKEN are provided.

CUDA_VERSION=${CUDA_VERSION:-12.5}
PYTHON_VERSION=${PYTHON_VERSION:-3.11}
GH_RUNNER_VERSION=${GH_RUNNER_VERSION:-2.317.0}
GH_RUNNER_DIR=${GH_RUNNER_DIR:-/opt/actions-runner}
GH_RUNNER_LABELS=${GH_RUNNER_LABELS:-"self-hosted,gpu"}
GH_RUNNER_NAME=${GH_RUNNER_NAME:-"$(hostname)-gpu"}

export DEBIAN_FRONTEND=noninteractive

if [[ $EUID -ne 0 ]]; then
  echo "This script must be run as root (use sudo)." >&2
  exit 1
fi

echo "Updating system packages..."
apt-get update
apt-get -y upgrade

echo "Installing base utilities..."
apt-get install -y \
  curl \
  wget \
  git \
  software-properties-common \
  build-essential \
  apt-transport-https \
  ca-certificates \
  gnupg \
  lsb-release

echo "Enabling Python ${PYTHON_VERSION} repository..."
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
apt-get install -y \
  python${PYTHON_VERSION} \
  python${PYTHON_VERSION}-dev \
  python3-pip

echo "Configuring CUDA repository (SBSA)..."
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/3bf863cc.pub | \
  gpg --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg arch=arm64] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/ /" \
  > /etc/apt/sources.list.d/cuda-sbsa.list
apt-get update
apt-get install -y cuda-toolkit-${CUDA_VERSION/./-}

if ! grep -q "CUDA" /etc/profile.d/cuda.sh 2>/dev/null; then
  cat <<'EOF' > /etc/profile.d/cuda.sh
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
EOF
fi

echo "Installing Docker..."
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "Installing NVIDIA Container Toolkit..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/${distribution}/$(dpkg --print-architecture) /" \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

echo "Installing Helm, Terraform, and Trivy..."
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
curl -fsSL https://apt.releases.hashicorp.com/gpg | gpg --dearmor -o /usr/share/keyrings/hashicorp.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp.gpg arch=$(dpkg --print-architecture)] https://apt.releases.hashicorp.com $(lsb_release -cs) main" \
  > /etc/apt/sources.list.d/hashicorp.list
curl -fsSL https://aquasecurity.github.io/trivy-repo/deb/public.key | gpg --dearmor -o /usr/share/keyrings/trivy.gpg
echo "deb [signed-by=/usr/share/keyrings/trivy.gpg] https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -cs) main" \
  > /etc/apt/sources.list.d/trivy.list
apt-get update
apt-get install -y terraform trivy

echo "Upgrading pip and installing Python tooling..."
python${PYTHON_VERSION} -m pip install --upgrade pip setuptools wheel virtualenv

if [[ -n "${GH_RUNNER_URL:-}" && -n "${GH_RUNNER_TOKEN:-}" ]]; then
  echo "Configuring GitHub Actions runner service..."
  mkdir -p "${GH_RUNNER_DIR}"
  cd "${GH_RUNNER_DIR}"

  if [[ ! -f "./config.sh" ]]; then
    ARCH=$(uname -m)
    case "${ARCH}" in
      x86_64) RUNNER_ARCH="x64" ;;
      aarch64|arm64) RUNNER_ARCH="arm64" ;;
      *)
        echo "Unsupported architecture for GitHub Actions runner: ${ARCH}" >&2
        exit 1
        ;;
    esac
    curl -fsSL -o actions-runner.tar.gz \
      "https://github.com/actions/runner/releases/download/v${GH_RUNNER_VERSION}/actions-runner-linux-${RUNNER_ARCH}-${GH_RUNNER_VERSION}.tar.gz"
    tar xzf actions-runner.tar.gz
    rm -f actions-runner.tar.gz
  fi

  if [[ ! -f ".runner" ]]; then
    ./config.sh --unattended \
      --url "${GH_RUNNER_URL}" \
      --token "${GH_RUNNER_TOKEN}" \
      --labels "${GH_RUNNER_LABELS}" \
      --name "${GH_RUNNER_NAME}"
  else
    echo "Runner already configured at ${GH_RUNNER_DIR}; skipping config."
  fi

  ./svc.sh install || true
  ./svc.sh start || true
  cd -
else
  cat <<'EOF'
GitHub Actions runner not configured.
Set GH_RUNNER_URL and GH_RUNNER_TOKEN (and optionally GH_RUNNER_LABELS, GH_RUNNER_NAME, GH_RUNNER_DIR)
before invoking this script to enable automatic runner registration.
EOF
fi

echo "GPU runner setup complete. Reboot recommended."

