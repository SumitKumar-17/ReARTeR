#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/lib/ollama/mlx_cuda_v13
PYTHON=~/miniconda3/envs/rearter/bin/python
SCRIPT=~/ReARTeR/analysis/run_scale_analysis.py
LOG_DIR=~/ReARTeR/analysis/results

echo '[QUEUE] Waiting for 50Q to finish...'
wait $(pgrep -f 'run_scale_analysis.*50')

for SCALE in 100 300 500; do
    echo ""
    echo "[QUEUE] Starting ${SCALE}Q — $(date)"
    $PYTHON $SCRIPT --scale $SCALE --corpus synth > $LOG_DIR/scale_${SCALE}_run.log 2>&1
    echo "[QUEUE] ${SCALE}Q done — $(date)"
done

echo ''
echo '[QUEUE] ALL DONE — '$(date)
