#!/bin/bash

BRANCH_NAME=$1
# Logic to handle optional venv path vs target dir
if [[ "$2" == /* ]] || [[ "$2" == .* ]]; then
    VENV_BASE_DIR=$2
    TARGET_DIR="terratorch.$BRANCH_NAME"
else
    TARGET_DIR=${2:-"terratorch.$BRANCH_NAME"}
    VENV_BASE_DIR=$3
fi

# Define the Python binary to use (pointing to your 3.12 install)
PYTHON_BIN="/dccstor/terratorch/python3.12.3/bin/python3.12"

# Fallback if the specific dccstor path doesn't exist
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN=$(which python3.10 2>/dev/null || which python3 2>/dev/null)
fi

BASE_PATH=$(pwd)

# 1. Validation ---
if [ -z "$BRANCH_NAME" ]; then
    echo "Usage: $0 <branch_name> [venv_base_path]"
    exit 1
fi

# 2. Safety Check ---
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Error: You are currently inside a git repository."
    exit 1
fi

# 3. Path Setup ---
mkdir -p "$TARGET_DIR"
FULL_PATH=$(cd "$TARGET_DIR" && pwd)

if [ -n "$VENV_BASE_DIR" ]; then
    mkdir -p "$VENV_BASE_DIR"
    VENV_ROOT=$(cd "$VENV_BASE_DIR" && pwd)
    VENV_PATH="$VENV_ROOT/venv_$BRANCH_NAME"
else
    VENV_PATH="$FULL_PATH/.venv"
fi

# 4. Clone / Checkout Logic ---
echo "Cloning Branch: $BRANCH_NAME into $FULL_PATH ---"
if [ ! -d "$FULL_PATH/.git" ]; then
    git clone git@github.com:torchgeo/terratorch.git "$FULL_PATH"
fi

cd "$FULL_PATH" || exit
git fetch origin
git checkout "$BRANCH_NAME" || git checkout -b "$BRANCH_NAME" "origin/$BRANCH_NAME"

# 5. Virtual Environment Setup ---
echo "Using Python: $($PYTHON_BIN --version) from $PYTHON_BIN"
echo "Setting up Virtual Environment at: $VENV_PATH ---"

# Clean up old venv if it was created with the wrong python version
if [ -d "$VENV_PATH" ]; then
    echo "Cleaning up existing environment..."
    rm -rf "$VENV_PATH"
fi

$PYTHON_BIN -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"

echo "Installing Dependencies (this may take a minute) ---"
pip install --upgrade pip
if ! pip install -e ".[test]"; then
    echo "Error: pip install failed. Ensure the python version is >= 3.10"
    exit 1
fi

# 6. LSF Submission ---
echo "Submitting to LSF (bsub) ---"
bsub -gpu "num=1" -Is -R "rusage[ngpus=1, cpu=4, mem=128GB]" \
     -J "terratorch_ci_$BRANCH_NAME" \
     "/bin/bash -c 'source $VENV_PATH/bin/activate && cd $FULL_PATH && pytest ./tests'"
