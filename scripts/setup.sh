#!/usr/bin/env bash
set -Eeuo pipefail

log() {
  printf '[setup] %s\n' "$1"
}

fail() {
  printf '[setup] ERROR: %s\n' "$1" >&2
  exit 1
}

trap 'fail "setup failed at line $LINENO. Check the message above for the failing step."' ERR

log "Reading credentials"
read -rsp "GH_TOKEN: " ghtoken
echo
# read -rsp "HF_TOKEN: " HF_TOKEN
# echo

[[ -n "${ghtoken:-}" ]] || fail "GitHub token is required"
# [[ -n "${HF_TOKEN:-}" ]] || fail "Hugging Face token is required"

repo="https://github.com/ManifoldsTeam/woodbench.git"
repo_with_token="${repo/https:\/\//https:\/\/$ghtoken@}"

if [[ -d woodbench ]]; then
  fail "Directory woodbench already exists. Remove it or run the script from a clean folder."
fi

log "Cloning repository"
git clone "$repo_with_token"
cd woodbench
git checkout hotfix


log "Installing Python dependencies"
python -m pip install -r requirements.txt

# log "Installing Hugging Face CLI"
# python -m pip install -U "huggingface_hub[cli]"

# log "Logging in to Hugging Face"
# hf auth login --token "$HF_TOKEN"

# log "Downloading Task B dataset"
# mkdir -p data/task-B
# hf download ny0g507h3p/wood \
#   --repo-type dataset \
#   --local-dir data/task-B-group-split

log "Setup completed successfully"
