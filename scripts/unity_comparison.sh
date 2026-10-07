#!/usr/bin/env bash
#SBATCH --job-name=calibration-comparison
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=USR1@120
#
# Resource/account/partition/array/time requests are supplied to sbatch.
# Usage: unity_comparison.sh PREPARED_NPZ OUTPUT_DIRECTORY check|production
# COMPARISON_CHECK_SWEEPS overrides the default 20-sweep execution check.
# This file does not submit jobs. See print_unity_commands.sh for commands.
set -euo pipefail

if [[ $# != 3 || ( $3 != check && $3 != production ) ]]; then
    echo "Usage: $0 PREPARED_NPZ OUTPUT_DIRECTORY check|production" >&2
    exit 2
fi
: "${SLURM_JOB_ID:?Run as a Slurm batch job on a compute node}"
: "${SLURM_ARRAY_TASK_ID:?An array task ID is required}"
: "${SLURM_CPUS_PER_TASK:?Set --cpus-per-task explicitly}"

# sbatch runs a spool copy of this script; --chdir must name the repository.
# COMPARISON_REPO permits an explicit override for other submission wrappers.
repo=${COMPARISON_REPO:-$PWD}
cd "$repo"
python_bin=${COMPARISON_PYTHON:-"$repo/.venv/bin/python"}
[[ -x "$python_bin" ]] || { echo "Missing Python environment: $python_bin" >&2; exit 2; }
[[ -f "$1" ]] || { echo "Missing prepared experiment: $1" >&2; exit 2; }

# One process per chain. srun binds JAX to the allocated cores; cap nested BLAS.
export JAX_ENABLE_X64=true
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

args=(--prepared "$1" --output "$2" --task-id "$SLURM_ARRAY_TASK_ID")
if [[ $3 == check ]]; then
    check_sweeps=${COMPARISON_CHECK_SWEEPS:-20}
    [[ $check_sweeps =~ ^[1-9][0-9]*$ ]] || {
        echo "COMPARISON_CHECK_SWEEPS must be a positive integer" >&2; exit 2;
    }
    args+=(--check-sweeps "$check_sweeps")
fi
exec srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
    "$python_bin" -m bayesiancalibration.comparison run "${args[@]}"
