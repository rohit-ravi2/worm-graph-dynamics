#!/bin/bash
# Run the five L2 gate-control folds on the GPU in parallel, then aggregate.
cd "$(dirname "$0")/../.."
PY=${PY:-python}; mkdir -p results/gd/logs
export PYTHONUNBUFFERED=1 WORMLAB_DEVICE=cuda PYTORCH_ALLOC_CONF=expandable_segments:True
for f in 0 1 2 3 4; do
  $PY scripts/gd/run_L2_gates.py $f > results/gd/logs/L2_gates_fold$f.log 2>&1 &
done
wait
$PY scripts/gd/run_L2_gates.py aggregate 2>&1 | grep -v Warn > results/gd/logs/L2_gates_aggregate.log
echo L2_GATES_ALL_DONE
