#!/usr/bin/env bash
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "[INFO] Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

export HF_CACHE=/mnt/taskmaster1/scratch/hyejong/gv_genomeocean/.hf_cache
export HF_HOME=$HF_CACHE
export HF_DATASETS_CACHE=$HF_CACHE/datasets
export TRANSFORMERS_CACHE=$HF_CACHE/transformers
export XDG_CACHE_HOME=$HF_CACHE
export HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$WORK_DIR"

VER="v3"
SIZE="5kb"
SHARD_SIZE=60000
MODEL_TAG="kd_8M"
MODEL_DIR_BASE="$WORK_DIR/compression/kd_${VER}_8M_${SIZE}"

# 현재는 fold1 학습이 끝난 상태 기준으로 기본값을 1로 둠
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  FOLDS="${SLURM_ARRAY_TASK_ID}"
else
  FOLDS="${FOLDS:-1}"
fi
echo "[INFO] FOLDS to run: $FOLDS"

for F in $FOLDS; do
  python "$WORK_DIR/scripts/test.py" \
    --work_dir "$WORK_DIR" \
    --model_dir "$MODEL_DIR_BASE/fold$F" \
    --model_tag "$MODEL_TAG" \
    --data_version "$VER" \
    --size_tag "$SIZE" \
    --fold "$F" \
    --shard_size "$SHARD_SIZE"
done
