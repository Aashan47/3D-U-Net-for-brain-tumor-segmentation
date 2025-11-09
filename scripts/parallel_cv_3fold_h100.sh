#!/bin/bash
# 3-Fold cross-validation training for H100 with nohup background execution
# Optimized for 24-hour completion with comprehensive logging

DATA_PATH=""
CONFIG_FILE="configs/experiments/h100_24h.yaml"
OUTPUT_DIR="./outputs"
LOG_DIR="./logs"

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
        --log_dir)
            LOG_DIR="$2"
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

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "H100 3-Fold Cross-Validation Training"
echo "====================================="
echo "Data path: $DATA_PATH"
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo "Logs: $LOG_DIR"
echo ""

# Create modified config for 3-fold training
cat > configs/h100_3fold.yaml << EOF
_base_: experiments/h100_24h.yaml

training:
  batch_size: 8
  max_epochs: 240  # Increased epochs for 3-fold (more time per fold)
  
data:
  cache_rate: 0.5  # Higher caching with only 3 folds
  
validation:
  patience: 30  # Increased patience for longer training
EOF

# Function to train a single fold with comprehensive logging
train_fold() {
    local fold=$1
    local log_file="$LOG_DIR/fold_${fold}_training.log"
    
    echo "Starting training for fold $fold with logging to $log_file"
    
    # Use nohup for background execution with virtual environment
    nohup bash -c "
        source venv/bin/activate
        python train_hdf5_real.py \
            --data_path '$DATA_PATH' \
            --fold $fold \
            --epochs 200 \
            --output_dir '$OUTPUT_DIR/fold_$fold'
    " > "$log_file" 2>&1 &
    
    local pid=$!
    echo "Fold $fold started with PID: $pid"
    echo $pid > "$LOG_DIR/fold_${fold}.pid"
    
    return $pid
}

# Function to train two folds in parallel
train_fold_pair() {
    local fold1=$1
    local fold2=$2
    
    echo "Starting parallel training: Fold $fold1 and Fold $fold2"
    
    # Start both folds
    train_fold $fold1
    local pid1=$!
    
    # Small delay to avoid resource conflicts
    sleep 30
    
    train_fold $fold2
    local pid2=$!
    
    echo "Training started:"
    echo "  Fold $fold1 - PID: $pid1 - Log: $LOG_DIR/fold_${fold1}_training.log"
    echo "  Fold $fold2 - PID: $pid2 - Log: $LOG_DIR/fold_${fold2}_training.log"
    
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
        echo "✗ Fold pair ($fold1, $fold2) failed (exit codes: $exit1, $exit2)"
        return 1
    fi
}

# Create master log file
MASTER_LOG="$LOG_DIR/training_master.log"
echo "3-Fold Cross-Validation Training Started: $(date)" > "$MASTER_LOG"
echo "Data: $DATA_PATH" >> "$MASTER_LOG"
echo "Config: $CONFIG_FILE" >> "$MASTER_LOG"
echo "Output: $OUTPUT_DIR" >> "$MASTER_LOG"
echo "Logs: $LOG_DIR" >> "$MASTER_LOG"
echo "===============================================" >> "$MASTER_LOG"

# Execute 3-fold training
START_TIME=$(date +%s)

echo "Phase 1: Training folds 0 and 1 in parallel"
echo "Phase 1: Training folds 0 and 1 in parallel" >> "$MASTER_LOG"
train_fold_pair 0 1

echo "Phase 2: Training fold 2"
echo "Phase 2: Training fold 2" >> "$MASTER_LOG"
train_fold 2
wait $!

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_HOURS=$((ELAPSED / 3600))
ELAPSED_MINUTES=$(((ELAPSED % 3600) / 60))

echo ""
echo "H100 3-Fold Training Complete!"
echo "=============================="
echo "Total time: ${ELAPSED_HOURS}h ${ELAPSED_MINUTES}m"
echo "All 3 folds trained"

# Log completion
echo "Training completed: $(date)" >> "$MASTER_LOG"
echo "Total time: ${ELAPSED_HOURS}h ${ELAPSED_MINUTES}m" >> "$MASTER_LOG"

# Create training status summary
cat > "$LOG_DIR/training_status.txt" << EOF
3-Fold Cross-Validation Training Status
=======================================
Started: $(date -d @$START_TIME)
Completed: $(date -d @$END_TIME)
Duration: ${ELAPSED_HOURS}h ${ELAPSED_MINUTES}m

Fold Logs:
- Fold 0: $LOG_DIR/fold_0_training.log
- Fold 1: $LOG_DIR/fold_1_training.log  
- Fold 2: $LOG_DIR/fold_2_training.log

Master Log: $LOG_DIR/training_master.log

Output Directories:
- Fold 0: $OUTPUT_DIR/fold_0
- Fold 1: $OUTPUT_DIR/fold_1
- Fold 2: $OUTPUT_DIR/fold_2

Next Steps:
1. Check individual fold logs for training progress
2. Run evaluation: python scripts/evaluate.py --data_path "$DATA_PATH" --output_dir "$OUTPUT_DIR"
3. Generate final report: python scripts/generate_report.py --output_dir "$OUTPUT_DIR"
EOF

# Cleanup
rm -f configs/h100_3fold.yaml

echo ""
echo "Training logs available at:"
echo "- Master log: $LOG_DIR/training_master.log"
echo "- Fold 0 log: $LOG_DIR/fold_0_training.log"
echo "- Fold 1 log: $LOG_DIR/fold_1_training.log"
echo "- Fold 2 log: $LOG_DIR/fold_2_training.log"
echo "- Status summary: $LOG_DIR/training_status.txt"
echo ""
echo "Monitor training progress with:"
echo "  tail -f $LOG_DIR/fold_0_training.log"
echo "  tail -f $LOG_DIR/fold_1_training.log"
echo "  tail -f $LOG_DIR/fold_2_training.log"