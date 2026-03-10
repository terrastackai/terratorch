#!/bin/bash

# Parse arguments
BRANCH_NAME=""
VENV_BASE_DIR=""
TARGET_DIR=""
SKIP_CLEANUP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cleanup)
            SKIP_CLEANUP=true
            shift
            ;;
        *)
            if [ -z "$BRANCH_NAME" ]; then
                BRANCH_NAME=$1
            elif [[ "$1" == /* ]] || [[ "$1" == .* ]]; then
                VENV_BASE_DIR=$1
                TARGET_DIR="terratorch.$BRANCH_NAME"
            else
                TARGET_DIR=$1
            fi
            shift
            ;;
    esac
done

# Set default TARGET_DIR if not specified
if [ -z "$TARGET_DIR" ]; then
    TARGET_DIR="terratorch.$BRANCH_NAME"
fi

# Agnostic Python Discovery
PYTHON_BIN=$(which /dccstor/terratorch/python3.12.3/bin/python3.12 2>/dev/null || which python3.10 2>/dev/null || which python3)

# Path relative to the repository root
TEST_FILE_PATH="integrationtests/test_base_set.py"

# 1. Validation ---
if [ -z "$BRANCH_NAME" ]; then
    echo "Usage: $0 <branch_name> [target_dir] [venv_base_path] [--no-cleanup]"
    echo "  --no-cleanup: Skip running the cleanup test"
    exit 1
fi

# 2. Path Setup ---
BASE_PATH=$(pwd)
mkdir -p "$TARGET_DIR"
FULL_PATH=$(cd "$TARGET_DIR" && pwd)

if [ -n "$VENV_BASE_DIR" ]; then
    mkdir -p "$VENV_BASE_DIR"
    VENV_ROOT=$(cd "$VENV_BASE_DIR" && pwd)
    VENV_PATH="$VENV_ROOT/venv_$BRANCH_NAME"
else
    VENV_PATH="$FULL_PATH/.venv"
fi

# 3. Clone & Checkout ---
if [ ! -d "$FULL_PATH/.git" ]; then
    echo "Cloning Terratorch into $FULL_PATH ---"
    git clone git@github.com:terrastackai/terratorch.git "$FULL_PATH"
fi

cd "$FULL_PATH" || exit
git fetch origin
git checkout "$BRANCH_NAME" || git checkout -b "$BRANCH_NAME" "origin/$BRANCH_NAME"

# 4. Environment Setup ---
if [ ! -d "$VENV_PATH" ]; then
    echo "Setting up Virtual Environment using $PYTHON_BIN..."
    $PYTHON_BIN -m venv "$VENV_PATH"
    source "$VENV_PATH/bin/activate"
    pip install --upgrade pip
    pip install -e ".[test]"
else
    source "$VENV_PATH/bin/activate"
fi

# 5. Extract Tests from root-level integrationtests folder ---
if [ ! -f "$TEST_FILE_PATH" ]; then
    echo "Error: Test file not found at $FULL_PATH/$TEST_FILE_PATH"
    echo "Current directory content:"
    ls -F
    exit 1
fi

# Pull test names (agnostic to whitespace/tabs)
TEST_LIST=$(grep -E '^def test_' "$TEST_FILE_PATH" | sed 's/def //g' | cut -d'(' -f1 | tr -d ' ')

# 6. Submit Individual Jobs ---
LOG_DIR="$FULL_PATH/lsf_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Found $(echo "$TEST_LIST" | wc -l) tests. Submitting individual jobs..."

# Separate test_models_fit from other tests
MODELS_FIT_TEST=""
OTHER_TESTS=""

for TEST_NAME in $TEST_LIST; do
    if [[ "$TEST_NAME" == "test_models_fit" ]]; then
        MODELS_FIT_TEST="$TEST_NAME"
    else
        OTHER_TESTS="$OTHER_TESTS $TEST_NAME"
    fi
done

# Define tests that depend on test_models_fit (require its checkpoints)
DEPENDENT_TESTS="test_latest_terratorch_version_buildings_predict test_latest_terratorch_version_floods_predict test_latest_terratorch_version_burnscars_predict"

# Categorize OTHER_TESTS as dependent, independent, or cleanup
DEPENDENT_TEST_LIST=""
INDEPENDENT_TEST_LIST=""
CLEANUP_TEST=""

for TEST_NAME in $OTHER_TESTS; do
    if [[ "$TEST_NAME" == "test_cleanup" ]]; then
        CLEANUP_TEST="$TEST_NAME"
    elif echo "$DEPENDENT_TESTS" | grep -qw "$TEST_NAME"; then
        DEPENDENT_TEST_LIST="$DEPENDENT_TEST_LIST $TEST_NAME"
    else
        INDEPENDENT_TEST_LIST="$INDEPENDENT_TEST_LIST $TEST_NAME"
    fi
done

# Track dependent job IDs for cleanup dependency
DEPENDENT_JOB_IDS=""

# Submit test_models_fit first (if it exists)
if [ -n "$MODELS_FIT_TEST" ]; then
    echo "Submitting test_models_fit (required prerequisite for dependent tests)..." >&2
    JOB_NAME="tt_${USER}_${MODELS_FIT_TEST}"
    
    MODELS_FIT_JOB_ID=$(bsub -gpu "num=1:mode=exclusive_process" -R "rusage[cpu=8, mem=32GB]" \
         -J "$JOB_NAME" \
         -o "$LOG_DIR/${MODELS_FIT_TEST}.log" \
         -e "$LOG_DIR/${MODELS_FIT_TEST}.err" \
         "/bin/bash -c 'set -e; source $VENV_PATH/bin/activate && cd $FULL_PATH && pytest $TEST_FILE_PATH -k $MODELS_FIT_TEST; exit \$?'" | grep -oP 'Job <\K[0-9]+')
    
    echo "test_models_fit submitted with Job ID: $MODELS_FIT_JOB_ID" >&2
    
    # Submit dependent tests with dependency on test_models_fit SUCCESS
    if [ -n "$DEPENDENT_TEST_LIST" ]; then
        echo "Submitting $(echo $DEPENDENT_TEST_LIST | wc -w) dependent test(s) (will wait for test_models_fit):$DEPENDENT_TEST_LIST" >&2
        for TEST_NAME in $DEPENDENT_TEST_LIST; do
            JOB_NAME="tt_${USER}_${TEST_NAME}"
            
            JOB_ID=$(bsub -gpu "num=1" -R "rusage[cpu=8, mem=32GB]" \
                 -w "ended($MODELS_FIT_JOB_ID)" \
                 -J "$JOB_NAME" \
                 -o "$LOG_DIR/${TEST_NAME}.log" \
                 -e "$LOG_DIR/${TEST_NAME}.err" \
                 "/bin/bash -c 'set -e; source $VENV_PATH/bin/activate && cd $FULL_PATH && pytest $TEST_FILE_PATH -k $TEST_NAME; exit \$?'" | grep -oP 'Job <\K[0-9]+')
            DEPENDENT_JOB_IDS="$DEPENDENT_JOB_IDS $JOB_ID"
        done
    fi
    
    # Submit cleanup test last - waits for all dependent tests to complete (success or failure)
    if [ -n "$CLEANUP_TEST" ] && [ -n "$DEPENDENT_JOB_IDS" ] && [ "$SKIP_CLEANUP" = false ]; then
        echo "Submitting cleanup test (will wait for all dependent tests to complete)..." >&2
        JOB_NAME="tt_${USER}_${CLEANUP_TEST}"
        
        # Build dependency condition: wait for all dependent jobs to end (regardless of exit status)
        CLEANUP_DEPENDENCY=""
        for JID in $DEPENDENT_JOB_IDS; do
            if [ -z "$CLEANUP_DEPENDENCY" ]; then
                CLEANUP_DEPENDENCY="ended($JID)"
            else
                CLEANUP_DEPENDENCY="$CLEANUP_DEPENDENCY && ended($JID)"
            fi
        done
        
        bsub -gpu "num=1" -R "rusage[cpu=8, mem=32GB]" \
             -w "$CLEANUP_DEPENDENCY" \
             -J "$JOB_NAME" \
             -o "$LOG_DIR/${CLEANUP_TEST}.log" \
             -e "$LOG_DIR/${CLEANUP_TEST}.err" \
             "/bin/bash -c 'set -e; source $VENV_PATH/bin/activate && cd $FULL_PATH && pytest $TEST_FILE_PATH -k $CLEANUP_TEST; exit \$?'"
        
        echo "Cleanup test will run after all dependent tests complete" >&2
    elif [ "$SKIP_CLEANUP" = true ]; then
        echo "Skipping cleanup test (--no-cleanup flag set)" >&2
    fi
    
    # Submit independent tests immediately (no dependency)
    if [ -n "$INDEPENDENT_TEST_LIST" ]; then
        echo "Submitting $(echo $INDEPENDENT_TEST_LIST | wc -w) independent test(s) (run immediately):$INDEPENDENT_TEST_LIST" >&2
        for TEST_NAME in $INDEPENDENT_TEST_LIST; do
            JOB_NAME="tt_${USER}_${TEST_NAME}"
            
            bsub -gpu "num=1" -R "rusage[cpu=8, mem=32GB]" \
                 -J "$JOB_NAME" \
                 -o "$LOG_DIR/${TEST_NAME}.log" \
                 -e "$LOG_DIR/${TEST_NAME}.err" \
                 "/bin/bash -c 'set -e; source $VENV_PATH/bin/activate && cd $FULL_PATH && pytest $TEST_FILE_PATH -k $TEST_NAME; exit \$?'"
        done
    fi
else
    echo "Error: test_models_fit not found in test suite." >&2
    echo "Error: test_models_fit is a required prerequisite that creates checkpoints for dependent tests." >&2
    
    # Report what will be skipped
    if [ -n "$DEPENDENT_TEST_LIST" ]; then
        echo "Warning: Skipping $(echo $DEPENDENT_TEST_LIST | wc -w) dependent test(s) (require test_models_fit):$DEPENDENT_TEST_LIST" >&2
    fi
    
    # Submit only independent tests
    if [ -n "$INDEPENDENT_TEST_LIST" ]; then
        echo "Info: Submitting $(echo $INDEPENDENT_TEST_LIST | wc -w) independent test(s):$INDEPENDENT_TEST_LIST" >&2
        for TEST_NAME in $INDEPENDENT_TEST_LIST; do
            JOB_NAME="tt_${USER}_${TEST_NAME}"
            
            bsub -gpu "num=1" -R "rusage[cpu=8, mem=32GB]" \
                 -J "$JOB_NAME" \
                 -o "$LOG_DIR/${TEST_NAME}.log" \
                 -e "$LOG_DIR/${TEST_NAME}.err" \
                 "/bin/bash -c 'set -e; source $VENV_PATH/bin/activate && cd $FULL_PATH && pytest $TEST_FILE_PATH -k $TEST_NAME; exit \$?'"
        done
    else
        echo "Error: No independent tests found. Cannot proceed without test_models_fit." >&2
        exit 1
    fi
fi

echo "---"
echo "All tests submitted. Check logs in: $LOG_DIR"
echo "Monitor with: bjobs -J 'tt_${USER}_*'"
echo "Note: Dependent tests will only run if test_models_fit passes (exit code 0)"
if [ "$SKIP_CLEANUP" = true ]; then
    echo "Note: Cleanup test skipped (--no-cleanup flag set)"
elif [ -n "$CLEANUP_TEST" ]; then
    echo "Note: test_cleanup will run last, after all dependent tests complete (regardless of success/failure)"
fi
