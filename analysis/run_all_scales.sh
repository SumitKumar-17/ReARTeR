#!/bin/bash
# Run all scaling experiments sequentially
# Each writes results to README.md automatically

PYTHON=~/miniconda3/envs/rearter/bin/python
LOG_DIR=~/ReARTeR/analysis/results
SCRIPT=~/ReARTeR/analysis/run_scale_analysis.py

echo "======================================"
echo "ReARTeR Scale Experiments"
echo "Started: $(date)"
echo "======================================"

# Step 1: Prepare datasets (download TriviaQA + build BM25 index)
echo ""
echo "[STEP 1] Preparing datasets from HuggingFace TriviaQA..."
$PYTHON ~/ReARTeR/analysis/prepare_scale_dataset.py 2>&1 | tee $LOG_DIR/prepare_dataset.log
if [ $? -ne 0 ]; then
    echo "ERROR: Dataset preparation failed. Check $LOG_DIR/prepare_dataset.log"
    exit 1
fi
echo "Dataset preparation complete."

# Step 2: Run experiments at each scale
for SCALE in 50 100 300 500; do
    echo ""
    echo "======================================"
    echo "RUNNING: $SCALE Questions"
    echo "======================================"
    $PYTHON $SCRIPT --scale $SCALE --corpus trivia \
        2>&1 | tee $LOG_DIR/scale_${SCALE}_run.log
    echo "Scale $SCALE complete. Log: $LOG_DIR/scale_${SCALE}_run.log"
done

echo ""
echo "======================================"
echo "ALL EXPERIMENTS COMPLETE: $(date)"
echo "Results in: $LOG_DIR"
echo "README updated: ~/ReARTeR/analysis/README.md"
echo "======================================"
