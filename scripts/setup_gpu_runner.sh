#!/usr/bin/env bash
set -euo pipefail

# Usage: sudo ./scripts/setup_gpu_runner.sh
# Installs NVIDIA driver, CUDA toolkit, Docker + NVIDIA Container Toolkit,
# Helm, Terraform, Trivy, Python 3.11, and Python dependencies required to run CI tests.

CUDA_VERSION=${CUDA_VERSION:-12.5}
PYTHON_VERSION=${PYTHON_VERSION:-3.11}

if [[ $EUID -ne 0 ]]; then
  echo "This script must be run as root (use sudo)." >&2
  exit 1
fi

echo "Updating system packages..."
apt-get update
apt-get -y upgrade

echo "Installing build essentials and utilities..."
apt-get install -y \
  curl \
  wget \
  git \
  software-properties-common \
  build-essential \
  apt-transport-https \
  ca-certificates \
  gnupg \
  lsb-release \
  python${PYTHON_VERSION} \
  python${PYTHON_VERSION}-dev \
  python3-pip

echo "Configuring CUDA repository..."
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/3bf863cc.pub | gpg --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg arch=arm64] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/ /" \
  > /etc/apt/sources.list.d/cuda-sbsa.list
apt-get update
apt-get install -y cuda-toolkit-${CUDA_VERSION/./-}

echo "Configuring environment variables..."
if ! grep -q "CUDA" /etc/profile.d/cuda.sh 2>/dev/null; then
  echo 'export PATH=/usr/local/cuda/bin:$PATH' > /etc/profile.d/cuda.sh
  echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> /etc/profile.d/cuda.sh
fi

echo "Installing Docker..."
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "Installing NVIDIA Container Toolkit..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/$(. /etc/os-release; echo $ID$VERSION_ID)/$(dpkg --print-architecture) /" \
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

echo "Upgrading pip and installing Python dependencies..."
python${PYTHON_VERSION} -m pip install --upgrade pip
python${PYTHON_VERSION} -m pip install --upgrade \
  virtualenv \
  setuptools \
  wheel

echo "GPU runner setup complete. Reboot recommended."

