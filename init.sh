#!/bin/bash
#SBATCH --job-name=tools_install
#SBATCH --output=install.out
#SBATCH --error=install.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=34
#SBATCH --mem=256G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=pi_tpoggio

set -euo pipefail

# ================= Versions =================
TORCH_VER="2.10.0"
TORCHVISION_VER="0.25.0"  # Must match torch version (torch 2.10 → torchvision 0.25)
VLLM_VER="0.13.0"
OPENRLHF_VER="0.9.1.post1"

# ================= Modules =================
module purge
module load StdEnv 2>/dev/null || true
module load gcc/12.2.0 2>/dev/null || true
# PyTorch cu128 is built against CUDA 12.8; nvcc major must match 12 or extension
# builds (flash-attn) abort. This site has cuda/12.9.1 (OK: same major as 12.8).
module load cuda/12.9.1

export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export PATH="$CUDA_HOME/bin:$PATH"

echo "CUDA_HOME=$CUDA_HOME"
nvcc --version
gcc --version | head -1
ldd --version | head -1



# ================= Build env =================
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export UV_CACHE_DIR="$TMPDIR/uv-cache"
export UV_LINK_MODE=copy

export MAX_JOBS="${SLURM_CPUS_PER_TASK:-34}"
export CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS"
export MAKEFLAGS="-j$MAX_JOBS"

export TORCH_CUDA_ARCH_LIST="8.0"

uv venv

uv pip install setuptools wheel ninja packaging

# install torch first (cu128 wheels; use 12.x nvcc above, not CUDA 13 toolkit)
uv pip install "torch==${TORCH_VER}" --index-url https://download.pytorch.org/whl/cu128
uv pip install psutil


# ================= flash-attn (build separately) =================
echo "Building flash-attn"
export MAX_JOBS=24
export NVCC_THREADS=1
export FLASH_ATTENTION_FORCE_BUILD=TRUE
# Match PyTorch's CXX11 ABI to avoid undefined symbol errors at runtime
export FLASH_ATTENTION_FORCE_CXX11_ABI=TRUE


uv pip install -v --no-build-isolation "flash-attn==2.8.3"


echo "Installing into venv at $(pwd)"
source .venv/bin/activate

uv pip install  -v vllm
python -c "import vllm; print('vllm OK:', vllm.__version__)"

uv pip install -r requirements.txt