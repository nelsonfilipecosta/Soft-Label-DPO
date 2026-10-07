#!/usr/bin/env bash
# One-time setup for a fresh GPU pod (RunPod PyTorch template). Safe to re-run.
#
# Usage:
#   1. On the pod:  git clone -b <branch> https://github.com/nelsonfilipecosta/Soft-Label-DPO.git /workspace/Soft-Label-DPO
#   2. On the Mac:  scp -P <port> -i ~/.ssh/id_ed25519 .env root@<ip>:/workspace/Soft-Label-DPO/.env
#   3. On the pod:  bash /workspace/Soft-Label-DPO/scripts/setup_pod.sh
#   4. Open a new shell (or `source ~/.bashrc`) so the environment below is loaded.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${WORKSPACE:-/workspace}"  # volume disk: survives a pod stop, unlike the container disk
MIN_UV_VERSION="0.11.24"              # the uv version that wrote uv.lock
MODEL_ID="allenai/OLMo-2-0425-1B-SFT"

step() { printf '\n==> %s\n' "$*"; }
fail() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
version_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]; }

cd "$REPO_DIR"

step "Loading secrets from .env"
[ -f .env ] || fail ".env not found in $REPO_DIR. Copy it from your Mac with scp (see the header of this script)."
set -a; . ./.env; set +a
[ -n "${HF_TOKEN:-}" ] || fail "HF_TOKEN is empty in .env"
[ -n "${WANDB_API_KEY:-}" ] || fail "WANDB_API_KEY is empty in .env"

# Caches live on the volume disk so models and packages are not downloaded again after a stop.
export HF_HOME="${HF_HOME:-$WORKSPACE/hf_cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORKSPACE/.cache/uv}"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR"

step "Installing tmux"
if command -v tmux >/dev/null; then
    echo "already installed"
else
    apt-get update -qq && apt-get install -y -qq tmux
fi

step "Installing uv"
curl -LsSf https://astral.sh/uv/install.sh | sh
# Puts ~/.local/bin first on the PATH, ahead of any uv that ships with the image.
. "$HOME/.local/bin/env"
UV_VERSION="$(uv --version | awk '{print $2}')"
echo "using $(command -v uv), version $UV_VERSION"
version_ge "$UV_VERSION" "$MIN_UV_VERSION" || fail "uv $UV_VERSION is older than $MIN_UV_VERSION, which wrote uv.lock"

step "Installing locked dependencies"
# --locked fails instead of silently re-resolving if uv.lock is out of date with pyproject.toml.
uv sync --locked

step "Checking GPU, Hugging Face and W&B"
uv run --no-sync python - <<'PY'
import sys

import torch
import transformers
import trl

print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), transformers {transformers.__version__}, trl {trl.__version__}")

if not torch.cuda.is_available():
    sys.exit("CUDA is not available. The pod's driver is probably too old for this torch build: "
             "redeploy with a newer CUDA version filter.")
props = torch.cuda.get_device_properties(0)
print(f"GPU: {props.name}, {props.total_memory / 1e9:.0f} GB")
if not torch.cuda.is_bf16_supported():
    sys.exit("This GPU has no bf16 support. Rent an Ampere-or-newer GPU.")

# A real bf16 matmul catches driver problems that is_available() misses.
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
assert torch.isfinite(x @ x).all()
print("bf16 matmul on GPU: ok")

from huggingface_hub import whoami
print(f"Hugging Face user: {whoami()['name']}")

import wandb
wandb.login(verify=True)
print("W&B key: ok")
PY

step "Downloading $MODEL_ID into $HF_HOME"
uv run --no-sync python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL_ID'))"

step "Writing environment to ~/.bashrc"
MARKER="# >>> soft-label-dpo >>>"
if grep -qF "$MARKER" "$HOME/.bashrc" 2>/dev/null; then
    echo "already present"
else
    cat >> "$HOME/.bashrc" <<EOF

$MARKER
. "\$HOME/.local/bin/env"
if [ -f "$REPO_DIR/.env" ]; then set -a; . "$REPO_DIR/.env"; set +a; fi
export HF_HOME="\${HF_HOME:-$HF_HOME}"
export UV_CACHE_DIR="\${UV_CACHE_DIR:-$UV_CACHE_DIR}"
# <<< soft-label-dpo <<<
EOF
    echo "added"
fi

step "Done in ${SECONDS}s"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "Next: open a new shell, start tmux, cd $REPO_DIR"
