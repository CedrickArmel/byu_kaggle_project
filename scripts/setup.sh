#!/bin/bash

set -e  # Exit immediately on error

# 🛠 System update and install sudo
apt-get update && apt-get upgrade -y
apt-get install sudo -y

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
export PJRT_DEVICE=TPU
export XLA_USE_BF16=1
export XLA_DOWNCAST_BF16=1
export XLA_TENSOR_ALLOCATOR_MAXSIZE=100000000
export TPU_ACCELERATOR_TYPE=v3-8

# Pyenv setup
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"
eval "$(pyenv virtualenv-init -)"

# Poetry in PATH
export PATH="/root/.local/bin:$PATH"

EOF
echo "✅ TPU environment setup complete. Reload your shell with: source ~/.bashrc"

# 🗝 SSH key generation
cat << 'EOF' >> ~/.ssh/config
Host github.com
  AddKeysToAgent yes
  UseKeychain yes
  IdentityFile /kaggle/working/.ssh/id_ed25519
  IgnoreUnknown UseKeychain
EOF

echo "🔄 Reloading shell..."
source ~/.bashrc