# 🗝 SSH key generation
echo "🔄 Seting up SSH key ..."

cat << 'EOF' >> ~/.bashrc

# SSH agent
eval "$(ssh-agent -s)"

EOF

echo "🔄 Reloading shell..."
source ~/.bashrc

cat << 'EOF' >> ~/.ssh/config

Host github.com
  AddKeysToAgent yes
  IdentityFile /kaggle/working/.ssh/id_ed25519

EOF

echo "🔄 Securing SSH key ..."
chmod 600 /kaggle/working/.ssh/id_ed25519
ssh-add /kaggle/working/.ssh/id_ed25519

git config --global user.name "Cédrick-Armel YEBOUET"
git config --global user.email "35418979+CedrickArmel@users.noreply.github.com"

echo "🔄 Reloading shell..."
source ~/.bashrc

echo "✅ Setup completed!"