#!/usr/bin/env bash

# Stop on error
set -e

# ==============
# Install a venv
# ==============
python3 -m venv ~/.local/few-venv
source ~/.local/few-venv/bin/activate

# ===============
# Fix permissions
# ===============
sudo chown -R few:few /workspaces

# ==================
# Install pre-commit
# ==================
pip install --no-cache-dir pre-commit

# ====================
# Configure pre-commit
# ====================
pre-commit install \
  --hook-type commit-msg \
  --hook-type pre-commit \
  --hook-type pre-push
