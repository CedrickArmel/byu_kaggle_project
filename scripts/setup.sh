#!/bin/bash

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

# Set environment variables for TPU
export ISTPUVM=1
export PJRT_DEVICE=TPU
export PT_XLA_DEBUG_LEVEL=1
export TF_CPP_MIN_LOG_LEVEL=2
export TPU_ACCELERATOR_TYPE=v3-8
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
export TPU_HOST_BOUNDS=1,1,1
export TPU_RUNTIME_METRICS_PORTS=8431,8432,8433,8434
export TPU_SKIP_MDS_QUERY=1
export TPU_WORKER_HOSTNAMES=localhost
export TPU_WORKER_ID=0
export XLA_TENSOR_ALLOCATOR_MAXSIZE=100000000

# Pyenv setup
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"
eval "$(pyenv virtualenv-init -)"

# Poetry in PATH
export PATH="/root/.local/bin:$PATH"

EOF
echo "✅ TPU environment setup complete. Reload your shell with: source ~/.bashrc"

echo "🔄 Reloading shell..."
source ~/.bashrc

echo "🔄 Creating virtual environment..."
pyenv install 3.10.16
pyenv virtualenv 3.10.16 byu_project

echo "✅ Setup completed!"
