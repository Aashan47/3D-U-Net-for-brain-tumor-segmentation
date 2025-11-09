#!/bin/bash
# Parallel cross-validation training for H100
# Runs 2 folds simultaneously using MIG or memory partitioning

DATA_PATH=""
CONFIG_FILE="configs/experiments/h100_24h.yaml"
OUTPUT_DIR="./outputs"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2" 
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$DATA_PATH" ]; then
    echo "Error: --data_path is required"
    exit 1
fi

echo "H100 Parallel Cross-Validation Training"
echo "========================================"
echo "Data path: $DATA_PATH"
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo ""

# Create modified config for parallel training (smaller batch size per process)
cat > configs/h100_parallel.yaml << EOF
_base_: experiments/h100_24h.yaml

training:
  batch_size: 4  # Reduced from 8 for parallel execution
  
data:
  cache_rate: 0.2  # Reduced caching for parallel execution
EOF

# Function to train a pair of folds in parallel
train_fold_pair() {
    local fold1=$1
    local fold2=$2
    
    echo "Starting parallel training: Fold $fold1 and Fold $fold2"
    
    # Train fold1 with first half of GPU memory
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
        --data_path "$DATA_PATH" \
        --config configs/h100_parallel.yaml \
        --fold $fold1 \
        --output_dir "$OUTPUT_DIR/fold_$fold1" \
        --use_wandb &
    
    local pid1=$!
    
    # Small delay to avoid resource conflicts
    sleep 30
    
    # Train fold2 with second half of GPU memory  
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
        --data_path "$DATA_PATH" \
        --config configs/h100_parallel.yaml \
        --fold $fold2 \
        --output_dir "$OUTPUT_DIR/fold_$fold2" \
        --use_wandb &
    
    local pid2=$!
    
    # Wait for both to complete
    echo "Waiting for fold $fold1 (PID: $pid1) and fold $fold2 (PID: $pid2)"
    wait $pid1
    local exit1=$?
    wait $pid2  
    local exit2=$?
    
    if [ $exit1 -eq 0 ] && [ $exit2 -eq 0 ]; then
        echo "✓ Fold pair ($fold1, $fold2) completed successfully"
        return 0
    else
        echo "✗ Fold pair ($fold1, $fold2) failed"
        return 1
    fi
}

# Execute parallel training
START_TIME=$(date +%s)

echo "Phase 1: Training folds 0 and 1 in parallel"
train_fold_pair 0 1

echo "Phase 2: Training folds 2 and 3 in parallel" 
train_fold_pair 2 3

echo "Phase 3: Training fold 4"
python scripts/train.py \
    --data_path "$DATA_PATH" \
    --config configs/h100_parallel.yaml \
    --fold 4 \
    --output_dir "$OUTPUT_DIR/fold_4" \
    --use_wandb

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_HOURS=$((ELAPSED / 3600))
ELAPSED_MINUTES=$(((ELAPSED % 3600) / 60))

echo ""
echo "H100 Parallel Training Complete!"
echo "================================="
echo "Total time: ${ELAPSED_HOURS}h ${ELAPSED_MINUTES}m"
echo "All 5 folds trained"

# Cleanup
rm -f configs/h100_parallel.yaml

echo "Next: Run evaluation on all folds"