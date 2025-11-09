#!/bin/bash
"""
Run 5-fold cross-validation training for BraTS segmentation.
"""

# Default parameters
DATA_PATH=""
CONFIG_FILE=""
OUTPUT_DIR="./outputs"
MAX_EPOCHS=300
BATCH_SIZE=2
LEARNING_RATE=1e-4
NUM_WORKERS=4
USE_WANDB=false
WANDB_PROJECT="brats-3d-segmentation"

# Parse command line arguments
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
        --max_epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --use_wandb)
            USE_WANDB=true
            shift
            ;;
        --wandb_project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --data_path PATH        Path to preprocessed BraTS data (required)"
            echo "  --config FILE           Configuration file path"
            echo "  --output_dir DIR        Output directory (default: ./outputs)"
            echo "  --max_epochs N          Maximum epochs (default: 300)"
            echo "  --batch_size N          Batch size (default: 2)"
            echo "  --learning_rate RATE    Learning rate (default: 1e-4)"
            echo "  --num_workers N         Number of workers (default: 4)"
            echo "  --use_wandb             Use Weights & Biases logging"
            echo "  --wandb_project NAME    W&B project name"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check required parameters
if [ -z "$DATA_PATH" ]; then
    echo "Error: --data_path is required"
    echo "Use --help for usage information"
    exit 1
fi

if [ ! -d "$DATA_PATH" ]; then
    echo "Error: Data path '$DATA_PATH' does not exist"
    exit 1
fi

# Print configuration
echo "BraTS 5-Fold Cross-Validation Training"
echo "======================================="
echo "Data path: $DATA_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Config file: ${CONFIG_FILE:-"default"}"
echo "Max epochs: $MAX_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Number of workers: $NUM_WORKERS"
echo "Use W&B: $USE_WANDB"
if [ "$USE_WANDB" = true ]; then
    echo "W&B project: $WANDB_PROJECT"
fi
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Store start time
START_TIME=$(date +%s)
echo "Starting cross-validation at $(date)"
echo ""

# Function to run training for a single fold
train_fold() {
    local fold=$1
    local fold_output_dir="$OUTPUT_DIR/fold_$fold"
    
    echo "Starting fold $fold training..."
    echo "Output directory: $fold_output_dir"
    
    # Build command
    cmd="python scripts/train.py"
    cmd="$cmd --data_path $DATA_PATH"
    cmd="$cmd --fold $fold"
    cmd="$cmd --output_dir $fold_output_dir"
    cmd="$cmd --max_epochs $MAX_EPOCHS"
    cmd="$cmd --batch_size $BATCH_SIZE"
    cmd="$cmd --learning_rate $LEARNING_RATE"
    cmd="$cmd --num_workers $NUM_WORKERS"
    
    if [ -n "$CONFIG_FILE" ]; then
        cmd="$cmd --config $CONFIG_FILE"
    fi
    
    if [ "$USE_WANDB" = true ]; then
        cmd="$cmd --use_wandb --wandb_project $WANDB_PROJECT"
    fi
    
    # Run training
    echo "Command: $cmd"
    echo ""
    
    if $cmd; then
        echo "✓ Fold $fold training completed successfully"
        return 0
    else
        echo "✗ Fold $fold training failed"
        return 1
    fi
}

# Run training for each fold
successful_folds=0
failed_folds=()

for fold in {0..4}; do
    echo "================================================"
    echo "FOLD $fold TRAINING"
    echo "================================================"
    
    if train_fold $fold; then
        ((successful_folds++))
        echo ""
    else
        failed_folds+=($fold)
        echo ""
        echo "Fold $fold failed. Continuing with next fold..."
        echo ""
    fi
done

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
ELAPSED_HOURS=$((ELAPSED_TIME / 3600))
ELAPSED_MINUTES=$(((ELAPSED_TIME % 3600) / 60))
ELAPSED_SECONDS=$((ELAPSED_TIME % 60))

# Print summary
echo "================================================"
echo "CROSS-VALIDATION SUMMARY"
echo "================================================"
echo "Total folds: 5"
echo "Successful folds: $successful_folds"
echo "Failed folds: ${#failed_folds[@]}"

if [ ${#failed_folds[@]} -gt 0 ]; then
    echo "Failed fold numbers: ${failed_folds[*]}"
fi

echo ""
echo "Elapsed time: ${ELAPSED_HOURS}h ${ELAPSED_MINUTES}m ${ELAPSED_SECONDS}s"
echo "Completed at: $(date)"
echo ""

# List output directories
echo "Output directories:"
for fold in {0..4}; do
    fold_dir="$OUTPUT_DIR/fold_$fold"
    if [ -d "$fold_dir" ]; then
        echo "  ✓ Fold $fold: $fold_dir"
    else
        echo "  ✗ Fold $fold: $fold_dir (not found)"
    fi
done

# Generate next steps
echo ""
echo "Next steps:"
echo "==========="
echo "1. Evaluate each fold:"
echo "   for fold in {0..4}; do"
echo "     python scripts/evaluate_fold.py \\"
echo "       --data_path $DATA_PATH \\"
echo "       --fold \$fold \\"
echo "       --model_path $OUTPUT_DIR/fold_\$fold/checkpoints/fold_best.pth \\"
echo "       --output_dir $OUTPUT_DIR/fold_\$fold/evaluation"
echo "   done"
echo ""
echo "2. Generate ensemble predictions:"
echo "   python scripts/ensemble_inference.py \\"
echo "     --model_paths $OUTPUT_DIR/fold_*/checkpoints/fold_best.pth \\"
echo "     --input_dir /path/to/test/data \\"
echo "     --output_dir $OUTPUT_DIR/ensemble_predictions"
echo ""
echo "3. Create explainability reports:"
echo "   python scripts/generate_explainability.py \\"
echo "     --input_dir /path/to/test/data \\"
echo "     --model_path $OUTPUT_DIR/fold_0/checkpoints/fold_best.pth \\"
echo "     --output_dir $OUTPUT_DIR/xai_analysis"
echo ""

# Exit with appropriate code
if [ $successful_folds -eq 5 ]; then
    echo "🎉 All folds completed successfully!"
    exit 0
else
    echo "⚠️  Some folds failed. Check logs for details."
    exit 1
fi