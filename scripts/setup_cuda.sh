#!/bin/bash

# TODO: replace the scripts by an unique makefile
set -e  # Exit immediately on error

# 🐍 Install pyenv
curl https://pyenv.run | bash

# 🧙 Install Poetry
curl -sSL https://install.python-poetry.org | python3 -


# 📂 Environment cleanup and TPU setup — append to ~/.bashrc
cat << 'EOF' >> ~/.bashrc

# Unset environment variables that are not needed
for var in MASTER_ADDR MASTER_PORT TPU_PROCESS_ADDRESSES XRT_TPU_CONFIG; do
    unset $var
done

export HYDRA_FULL_ERROR=1

# Set environment variables for GPU
export CUDA_HOME="/usr/local/cuda"
export CUDA_VERSION="12.5.1"
export CUDA_MAJOR_VERSION="12"
export CUDA_MINOR_VERSION="5"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64/stubs"
export LIBRARY_PATH="/usr/local/cuda/lib64/stubs"

export NVARCH="x86_64"

export NVIDIA_VISIBLE_DEVICES="all"
export NVIDIA_DRIVER_CAPABILITIES="compute,utility"

export NV_CUDA_CUDART_VERSION="12.5.82-1"
export NV_CUDA_CUDART_DEV_VERSION="12.5.82-1"
export NV_CUDA_LIB_VERSION="12.5.1-1"
export NV_CUDA_NSIGHT_COMPUTE_VERSION="12.5.1-1"
export NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE="cuda-nsight-compute-12-5=12.5.1-1"
export NV_NVTX_VERSION="12.5.82-1"
export NV_NVPROF_VERSION="12.5.82-1"
export NV_NVPROF_DEV_PACKAGE="cuda-nvprof-12-5=12.5.82-1"

# cuDNN
export NV_CUDNN_VERSION="9.2.1.18-1"
export NV_CUDNN_PACKAGE="libcudnn9-cuda-12=9.2.1.18-1"
export NV_CUDNN_PACKAGE_DEV="libcudnn9-dev-cuda-12=9.2.1.18-1"

# cuBLAS
export NV_LIBCUBLAS_VERSION="12.5.3.2-1"
export NV_LIBCUBLAS_PACKAGE="libcublas-12-5=12.5.3.2-1"
export NV_LIBCUBLAS_PACKAGE_NAME="libcublas-12-5"
export NV_LIBCUBLAS_DEV_VERSION="12.5.3.2-1"
export NV_LIBCUBLAS_DEV_PACKAGE="libcublas-dev-12-5=12.5.3.2-1"
export NV_LIBCUBLAS_DEV_PACKAGE_NAME="libcublas-dev-12-5"

# NCCL (for multi-GPU)
export NCCL_VERSION="2.22.3-1"
export NV_LIBNCCL_PACKAGE="libnccl2=2.22.3-1+cuda12.5"
export NV_LIBNCCL_PACKAGE_NAME="libnccl2"
export NV_LIBNCCL_PACKAGE_VERSION="2.22.3-1"
export NV_LIBNCCL_DEV_PACKAGE="libnccl-dev=2.22.3-1+cuda12.5"
export NV_LIBNCCL_DEV_PACKAGE_NAME="libnccl-dev"
export NV_LIBNCCL_DEV_PACKAGE_VERSION="2.22.3-1"

# Pyenv setup
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"
eval "$(pyenv virtualenv-init -)"

# Poetry in PATH
export PATH="/root/.local/bin:$PATH"

EOF
echo "✅ GPU environment setup complete. Reload your shell with: source ~/.bashrc"

echo "🔄 Reloading shell..."
source ~/.bashrc

echo "🔄 Creating virtual environment..."
pyenv install 3.10.16
pyenv virtualenv 3.10.16 byu_project

echo "✅ Setup completed!"
