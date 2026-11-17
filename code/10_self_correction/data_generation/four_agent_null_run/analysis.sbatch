#!/bin/bash
#SBATCH --account=<slurm-account>
#SBATCH --partition=standard
#SBATCH --job-name=mmo-analysis
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=<cluster-scratch>/multi-model-originals-20261030/logs/%x-%j.out
# Any of this run's analysis scripts, never on a login node. No model calls.
#   sbatch analysis.sbatch make_table.py <cluster-scratch>/multi-model-originals-20261030/results
set -euo pipefail
export MMO_ROOT=${MMO_ROOT:-<cluster-scratch>/multi-model-originals-20261030}
export PYTHONDONTWRITEBYTECODE=1
cd "$MMO_ROOT/code"
<cluster-software>/arc/python/3.12.1/bin/python3 "$@"
