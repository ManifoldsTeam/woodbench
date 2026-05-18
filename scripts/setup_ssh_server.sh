#!/bin/bash
# =============================================================================
# setup_ssh_server.sh — Setup woodbench on a remote server
#
# Flow:
#   1. Connect to vast.ai via SSH
#   2. Enter git token → clone the repo
#   3. Install project in editable mode (pip install -e .)
#   4. Install huggingface_hub
#   5. Enter HF token → login to HuggingFace + download dataset
#
# Usage:
#   bash scripts/setup_ssh_server.sh
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

REPO_URL="https://github.com/ManifoldsTeam/woodbench.git"
HF_TASK_A_REPO="ny0g507h3p/wood-task-A"
PROJECT_ROOT="/workspace/woodbench"
DATA_DIR="${PROJECT_ROOT}/data/task-A"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERR]${NC}  $*" >&2; }
log_step()  { echo -e "\n${CYAN}==== $* ====${NC}"; }
log_banner(){ echo -e "\n${BOLD}*** $* ***${NC}"; }

prompt_password() {
    local prompt="$1"
    local var_name="$2"
    local value
    read -rsp "$(echo -e "${CYAN}${prompt}${NC}: ")" value
    echo
    printf -p "$var_name" "%s" "$value"
}

# =============================================================================
# Step 1: Enter git token and clone repo
# =============================================================================

clone_repo() {
    log_step "Step 1 — Clone woodbench Repository"

    echo "  You need a GitHub token to clone the private repo."
    echo "  Get one at: https://github.com/settings/tokens"
    echo "  (Classification: 'repo' for private repos)"
    echo ""

    prompt_password "  GitHub token" GITHUB_TOKEN

    if [[ -z "$GITHUB_TOKEN" ]]; then
        log_error "GitHub token is required."
        exit 1
    fi

    log_info "Cloning woodbench to ${PROJECT_ROOT}..."

    local remote_url="${REPO_URL//https:\/\/github\.com\//https://${GITHUB_TOKEN}@github.com/}"
    git clone "$remote_url" "${PROJECT_ROOT}"

    log_info "Repository cloned."
}

# =============================================================================
# Step 2: Install project in editable mode
# =============================================================================

install_project() {
    log_step "Step 2 — Install woodbench Project"

    log_info "Installing project in editable mode..."
    pip install -e .

    log_info "Project installed."
}

# =============================================================================
# Step 3: Install huggingface_hub
# =============================================================================

install_huggingface_hub() {
    log_step "Step 3 — Install huggingface_hub"

    if command -v python &>/dev/null; then
        log_info "Python found: $(python --version)"
    else
        log_error "Python not found. Please install Python first."
        exit 1
    fi

    log_info "Installing huggingface_hub..."
    pip install huggingface_hub --quiet

    log_info "huggingface_hub installed."
}

# =============================================================================
# Step 4: Login to HuggingFace and download dataset
# =============================================================================

login_and_download() {
    log_step "Step 4 — Login to HuggingFace and Download Dataset"

    echo "  You need a HuggingFace token to download the dataset."
    echo "  Get one at: https://huggingface.co/settings/tokens"
    echo "  (Token type: 'Read' is sufficient)"
    echo ""

    prompt_password "  HuggingFace token" HF_TOKEN

    if [[ -z "$HF_TOKEN" ]]; then
        log_error "HuggingFace token is required."
        exit 1
    fi

    log_info "Logging in to HuggingFace..."
    python -c "
from huggingface_hub import login
login(token='${HF_TOKEN}')
"
    log_info "Logged in."

    log_info "Creating data directory: ${DATA_DIR}"
    mkdir -p "${DATA_DIR}"

    log_info "Downloading dataset ${HF_TASK_A_REPO}..."
    hf download \
        ny0g507h3p/wood-task-A \
        --repo-type dataset \
        --local-dir "${DATA_DIR}"

    log_info "Dataset ready at ${DATA_DIR}"
}

# =============================================================================
# Main
# =============================================================================

main() {
    echo -e "${BOLD}"
    echo "============================================================"
    echo "  woodbench — Server Setup"
    echo "============================================================"
    echo -e "${NC}"
    echo "  Project: ${PROJECT_ROOT}"
    echo "  Data:    ${DATA_DIR}"
    echo ""

    clone_repo

    echo ""
    cd "${PROJECT_ROOT}"
    log_info "Changed to project directory: ${PROJECT_ROOT}"

    install_project

    install_huggingface_hub

    login_and_download

    log_step "Setup Complete"
    echo "  Project:  ${PROJECT_ROOT}"
    echo "  Data:     ${DATA_DIR}"
    echo ""
    log_info "All done! You can now start working on your project."
}

main "$@"
