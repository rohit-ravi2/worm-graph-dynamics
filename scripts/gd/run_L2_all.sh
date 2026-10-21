#!/bin/bash
# Run the five L2 folds on the GPU in parallel, then aggregate and capsule.
cd "$(dirname "$0")/../.."
PY=${PY:-python}; mkdir -p results/gd/logs
export PYTHONUNBUFFERED=1 WORMLAB_DEVICE=cuda PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for f in 0 1 2 3 4; do
  [ -f results/gd/benchmarks/L2/fold$f.json ] && continue
  $PY scripts/gd/run_L2.py $f > results/gd/logs/L2_fold$f.log 2>&1 &
done
wait
n=$(ls results/gd/benchmarks/L2/fold*.json 2>/dev/null | wc -l)
if [ "$n" -lt 5 ]; then
  echo "FOLDS INCOMPLETE $n/5"; exit 1
fi
$PY scripts/gd/run_L2.py aggregate 2>&1 | grep -v Warn > results/gd/logs/L2_aggregate.log
$PY -m wormlab.cli gd run L2 2>&1 | grep -v Warn > results/gd/logs/L2_capsule.log
echo L2_ALL_DONE
