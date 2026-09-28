#!/usr/bin/env bash
# Print shell-escaped submission commands ONLY; never connect or call sbatch.
# Required environment: PARTITION, CPUS, MEMORY, CHECK_WALLTIME, PRODUCTION_WALLTIME.
# Optional: ACCOUNT, CONSTRAINT (use one CPU/node class), MAX_PARALLEL (default 20).
# Production walltime must cover compilation + 1000 warmup sweeps + 6h + I/O.
set -euo pipefail
if [[ $# != 2 ]]; then
    echo "Usage: $0 PREPARED_NPZ OUTPUT_DIRECTORY" >&2
    exit 2
fi
: "${PARTITION:?Specify the Unity partition}"
: "${CPUS:?Specify identical allocated CPU cores per chain}"
: "${MEMORY:?Specify per-job memory, e.g. a measured value in G}"
: "${CHECK_WALLTIME:?Specify the execution-check Slurm time limit}"
: "${PRODUCTION_WALLTIME:?Specify total time including warmup plus 6h production}"
if [[ ! $CPUS =~ ^[1-9][0-9]*$ || ! ${MAX_PARALLEL:-20} =~ ^[1-9][0-9]*$ ]]; then
    echo "CPUS and MAX_PARALLEL must be positive integers" >&2
    exit 2
fi
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
[[ -f "$1" ]] || { echo "Prepare the common experiment first: $1" >&2; exit 2; }
prepared=$(cd -- "$(dirname -- "$1")" && pwd)/$(basename -- "$1")
mkdir -p -- "$2/logs"
output=$(cd -- "$2" && pwd)
common=(sbatch --chdir="$repo" --partition="$PARTITION" --cpus-per-task="$CPUS"
    --mem="$MEMORY" --output="$output/logs/%x-%A_%a.out"
    --error="$output/logs/%x-%A_%a.err")
[[ -z ${ACCOUNT:-} ]] || common+=(--account="$ACCOUNT")
[[ -z ${CONSTRAINT:-} ]] || common+=(--constraint="$CONSTRAINT")
echo '# First: one chain per method; 20 warmup sweeps, no retained production.'
printf '%q ' "${common[@]}" --time="$CHECK_WALLTIME" --array=0,4,8,12,16 \
    "$repo/scripts/unity_comparison.sh" "$prepared" "$output" check
printf '\n'
echo '# After reviewing checks and resource/tuning feasibility: 20 independent jobs.'
printf '%q ' "${common[@]}" --time="$PRODUCTION_WALLTIME" \
    "--array=0-19%${MAX_PARALLEL:-20}" "$repo/scripts/unity_comparison.sh" \
    "$prepared" "$output" production
printf '\n'
echo '# Nothing was submitted. Reuse the same command/output directory to resume.'
