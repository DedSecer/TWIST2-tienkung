#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${VENV_PATH:-$ROOT_DIR/.venv-twist2-py38}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_PATH/bin/python}"
UV_BIN="${UV_BIN:-uv}"
PYTORCH_SRC_DIR="${PYTORCH_SRC_DIR:-$HOME/src/pytorch-sm120-py38}"
VISION_SRC_DIR="${VISION_SRC_DIR:-$HOME/src/vision-sm120-py38}"
PYTORCH_VERSION="${PYTORCH_VERSION:-v2.4.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-v0.19.1}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
INDEX_URL="${INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
MAX_JOBS_VALUE="${MAX_JOBS:-$(nproc)}"
TORCH_CUDA_ARCH_LIST_VALUE="${TORCH_CUDA_ARCH_LIST:-12.0}"

log() {
    printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1" >&2
        exit 1
    fi
}

check_file() {
    if [[ ! -e "$1" ]]; then
        echo "Missing required path: $1" >&2
        exit 1
    fi
}

log "Checking prerequisites"
require_cmd git
require_cmd "$UV_BIN"
require_cmd "$CUDA_HOME/bin/nvcc"
require_cmd gcc-11
require_cmd g++-11
require_cmd cmake
require_cmd ninja
require_cmd nvidia-smi
check_file "$PYTHON_BIN"

log "Printing toolchain summary"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
"$CUDA_HOME/bin/nvcc" --version | tail -n 1
"$CUDA_HOME/bin/nvcc" --list-gpu-arch | grep -qx 'compute_120' || {
    echo "nvcc does not report compute_120 support" >&2
    exit 1
}
gcc-11 --version | head -n 1
g++-11 --version | head -n 1
cmake --version | head -n 1
ninja --version
"$PYTHON_BIN" --version

mkdir -p "$HOME/src"

log "Installing Python build dependencies into $VENV_PATH"
"$UV_BIN" pip install --python "$PYTHON_BIN" -i "$INDEX_URL" \
    setuptools \
    wheel \
    setuptools_scm \
    typing-extensions \
    sympy \
    pyyaml \
    "numpy==1.23.0" \
    ninja \
    cmake \
    requests \
    six \
    future \
    filelock \
    jinja2 \
    networkx

log "Removing prebuilt torch packages from $VENV_PATH"
"$UV_BIN" pip uninstall --python "$PYTHON_BIN" torch torchvision triton || true

log "Preparing PyTorch source tree at $PYTORCH_SRC_DIR"
if [[ ! -d "$PYTORCH_SRC_DIR/.git" ]]; then
    git clone --recursive https://github.com/pytorch/pytorch.git "$PYTORCH_SRC_DIR"
fi
git -C "$PYTORCH_SRC_DIR" fetch --all --tags
git -C "$PYTORCH_SRC_DIR" checkout "$PYTORCH_VERSION"
git -C "$PYTORCH_SRC_DIR" submodule sync
git -C "$PYTORCH_SRC_DIR" submodule update --init --recursive

log "Installing PyTorch source requirements"
"$UV_BIN" pip install --python "$PYTHON_BIN" -r "$PYTORCH_SRC_DIR/requirements.txt" -i "$INDEX_URL"

export CC=gcc-11
export CXX=g++-11
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST_VALUE"
export MAX_JOBS="$MAX_JOBS_VALUE"
export CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS_VALUE"
export USE_CUDA=1
export USE_CUDNN=1
export USE_NCCL=1
export USE_DISTRIBUTED=0
export USE_KINETO=0
export BUILD_TEST=0
export USE_TRITON=0
export USE_FBGEMM=0
export PYTORCH_BUILD_VERSION="${PYTORCH_VERSION#v}"
export PYTORCH_BUILD_NUMBER=1
export CMAKE_PREFIX_PATH="$($PYTHON_BIN -c 'import site; print(site.getsitepackages()[0])')"

log "Building PyTorch wheel"
pushd "$PYTORCH_SRC_DIR" >/dev/null
"$PYTHON_BIN" setup.py bdist_wheel
TORCH_WHEEL="$(ls -t dist/torch-*.whl | head -n 1)"
"$UV_BIN" pip install --python "$PYTHON_BIN" --force-reinstall "$TORCH_WHEEL"
popd >/dev/null

log "Preparing torchvision source tree at $VISION_SRC_DIR"
if [[ ! -d "$VISION_SRC_DIR/.git" ]]; then
    git clone --recursive https://github.com/pytorch/vision.git "$VISION_SRC_DIR"
fi
git -C "$VISION_SRC_DIR" fetch --all --tags
git -C "$VISION_SRC_DIR" checkout "$TORCHVISION_VERSION"
git -C "$VISION_SRC_DIR" submodule sync
git -C "$VISION_SRC_DIR" submodule update --init --recursive

export FORCE_CUDA=1

log "Building torchvision wheel"
pushd "$VISION_SRC_DIR" >/dev/null
"$PYTHON_BIN" setup.py bdist_wheel
VISION_WHEEL="$(ls -t dist/torchvision-*.whl | head -n 1)"
"$UV_BIN" pip install --python "$PYTHON_BIN" --force-reinstall "$VISION_WHEEL"
popd >/dev/null

log "Clearing torch extension cache so Isaac Gym rebuilds gymtorch"
rm -rf "$HOME/.cache/torch_extensions"

log "Running CUDA smoke test"
"$PYTHON_BIN" - <<'PY'
import torch

print('torch:', torch.__version__)
print('cuda version:', torch.version.cuda)
print('is_available:', torch.cuda.is_available())
print('device_count:', torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit('CUDA is not available after build')

print('device_name:', torch.cuda.get_device_name(0))
x = torch.randn(16, device='cuda')
y = x * 2
print('cuda_tensor_ok:', y[:4])
PY

log "Running Isaac Gym import smoke test"
"$PYTHON_BIN" - <<'PY'
import torch
import isaacgym
from isaacgym import gymapi, gymtorch

print('isaacgym ok')
print('torch cuda available:', torch.cuda.is_available())
print('device_name:', torch.cuda.get_device_name(0))
PY

log "Done. You can now try: source .venv-twist2-py38/bin/activate && bash train.sh 0408_g1_cuda cuda:0"