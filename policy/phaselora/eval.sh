#!/bin/bash
# bash eval.sh <task_name> <setting> <checkpoint_path> [gpu_id] [extra_opts...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
EMBODICHAIN_ROOT="${EMBODICHAIN_ROOT:-$WORKSPACE_ROOT/EmbodiChain}"
PYTHON_BIN="${PYTHON_BIN:-python}"
POLICY_NAME=phaselora

TASK_NAME="${1}"
SETTING="${2}"
CHECKPOINT_PATH="${3}"
GPU_ID="${4:-0}"
shift 4 2>/dev/null || true
EXTRA_ARGS=("$@")

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/policy${PYTHONPATH:+:$PYTHONPATH}"
if [[ -d "$EMBODICHAIN_ROOT" ]]; then
    export PYTHONPATH="$EMBODICHAIN_ROOT:$PYTHONPATH"
fi
cd "$REPO_ROOT"

if [[ "$GPU_ID" == "1" ]]; then
    unset CUDA_VISIBLE_DEVICES
    DEVICE_ARGS=(--gpu_id 1 --pytorch_device cuda:1)
else
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    DEVICE_ARGS=(--gpu_id 0 --pytorch_device cuda:0)
fi

PYTHONWARNINGS=ignore::UserWarning \
EMBODICHAIN_SIM_EXIT_PROCESS=0 \
"$PYTHON_BIN" scripts/eval_policy.py \
    --config policy/$POLICY_NAME/deploy_policy.yml \
    --overrides \
    --task_name "$TASK_NAME" \
    --setting "$SETTING" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --model_name "$(basename "$CHECKPOINT_PATH")" \
    "${DEVICE_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
