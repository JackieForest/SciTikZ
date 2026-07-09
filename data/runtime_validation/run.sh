#!/bin/bash
set -euo pipefail

# Configuration: set BASE and PARTITION
: "${BASE:?ERROR: please export BASE path to distill directory}"
: "${PARTITION:?ERROR: please export PARTITION name for Slurm}"
# Example: export BASE="/path/to/distill/dir"
# Example: export PARTITION="your_partition_name"

PY="${BASE}/run.py"

LOG_DIR="${BASE}/logs_compile"
mkdir -p "${LOG_DIR}"

echo "Launching 1 parallel srun jobs..."
echo "BASE=${BASE}"
echo "PARTITION=${PARTITION}"
echo "PY=${PY}"
echo "LOG_DIR=${LOG_DIR}"

for i in {0..0}; do
  echo "Submitting split ${i} ..."

  nohup srun -p "${PARTITION}" \
    --quotatype=reserved \
    --job-name=compile_${i} \
    python "${PY}" --idx "${i}" \
      --base "${BASE}" \
      --only_ok \
      --shell_escape \
    > "${LOG_DIR}/compile_${i}.out" 2>&1 &

  sleep 3
done

echo "All submitted."
echo "Check logs: tail -f ${LOG_DIR}/compile_*.out"
