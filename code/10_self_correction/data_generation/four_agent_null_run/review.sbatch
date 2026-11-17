#!/bin/bash
#SBATCH --account=<slurm-account>
#SBATCH --partition=standard
#SBATCH --job-name=mmo-review
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=<cluster-scratch>/multi-model-originals-20261030/logs/%x-%A_%a.out
# Sol for every read:      sbatch --array=0-1 --export=ALL,MODEL=gpt-6.1-sol,ACCOUNT=primary,TASKS=2,WORKERS=10 review.sbatch
# Terra, Sol cross-check:  sbatch --array=0-1 --export=ALL,MODEL=gpt-6.1-terra,ACCOUNT=second,READ=1,TASKS=2,WORKERS=10 review.sbatch
#                          sbatch --array=0 --export=ALL,MODEL=gpt-6.1-sol,ACCOUNT=primary,READ=2,TASKS=1,WORKERS=10 review.sbatch
set -euo pipefail
export MMO_ROOT=${MMO_ROOT:-<cluster-scratch>/multi-model-originals-20261030}
export PYTHONDONTWRITEBYTECODE=1 MMO_PHASE_TIMEOUT=300 MMO_REVIEW=${MMO_REVIEW:-review}
export TMPDIR=/tmp/mmo-review-$SLURM_JOB_ID-${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
cd "$MMO_ROOT/code"
<cluster-software>/arc/python/3.12.1/bin/python3 review.py --model "$MODEL" --account "$ACCOUNT" ${READ:+--read "$READ"} --task "${SLURM_ARRAY_TASK_ID:-0}" --tasks "${TASKS:-1}" --workers "${WORKERS:-10}" ${LIMIT:+--limit "$LIMIT"}
