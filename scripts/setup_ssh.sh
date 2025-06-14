#!/bin/bash
# TODO: overite ./bashrc with my own things

# 🗝 SSH key generation
echo "🔄 Seting up SSH key ..."

cat << 'EOF' >> ~/.bashrc

# SSH agent
eval "$(ssh-agent -s)"

EOF

echo "🔄 Reloading shell..."
source ~/.bashrc

mkdir -p ~/.ssh
touch ~/.ssh/config

cat << 'EOF' >> ~/.ssh/config

Host github.com
  AddKeysToAgent yes
  IdentityFile /kaggle/input/ssh-keys/kaggle_ssh

EOF

echo "🔄 Securing SSH key ..."
chmod 600 /kaggle/input/ssh-keys/kaggle_ssh
ssh-add /kaggle/input/ssh-keys/kaggle_ssh

echo "🔄 Set git global config ..."
git config --global user.name "Cédrick-Armel YEBOUET"
git config --global user.email "35418979+CedrickArmel@users.noreply.github.com"

echo "🔄 Reloading shell..."
source ~/.bashrc

echo "✅ Setup completed!"
