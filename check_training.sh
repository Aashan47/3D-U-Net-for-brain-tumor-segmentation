#!/bin/bash
# Comprehensive BraTS Training Status Monitor
# Monitors dataset download, training progress, and system resources

LOG_DIR="/home/azureuser/3D-medical-image/brats-3d-segmentation/logs"
PROJECT_DIR="/home/azureuser/3D-medical-image/brats-3d-segmentation"

echo "🧠 BraTS 3D Segmentation - Complete Pipeline Status"
echo "=================================================="
echo "⏰ Timestamp: $(date)"
echo ""

# Function to check process status
check_process() {
    local pid_file=$1
    local process_name=$2
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        echo "📋 $process_name PID: $pid"
        
        if ps -p $pid > /dev/null 2>&1; then
            echo "✅ $process_name is RUNNING"
            return 0
        else
            echo "❌ $process_name is NOT running"
            return 1
        fi
    else
        echo "⚠️  No $process_name PID file found"
        return 1
    fi
}

# Check main pipeline process
echo "🔍 Process Status:"
echo "=================="
check_process "$LOG_DIR/pipeline.pid" "Main Pipeline"
check_process "$LOG_DIR/training.pid" "Training Process" 

# Check individual fold PIDs
for fold in 0 1 2; do
    if [ -f "$LOG_DIR/fold_${fold}.pid" ]; then
        check_process "$LOG_DIR/fold_${fold}.pid" "Fold $fold Training"
    fi
done

echo ""

# Check dataset status
echo "📊 Dataset Status:"
echo "=================="
DATA_DIR="$PROJECT_DIR/data/raw"
if [ -d "$DATA_DIR" ] && [ -n "$(find "$DATA_DIR" -name "BraTS*" -type d 2>/dev/null | head -1)" ]; then
    FOUND_DATA=$(find "$DATA_DIR" -name "BraTS*" -type d | head -1)
    PARENT_DIR=$(dirname "$FOUND_DATA")
    SUBJECT_COUNT=$(find "$PARENT_DIR" -name "BraTS*" -type d | wc -l)
    DATASET_SIZE=$(du -sh "$PARENT_DIR" 2>/dev/null | cut -f1)
    echo "✅ Dataset found: $PARENT_DIR"
    echo "📁 Subjects: $SUBJECT_COUNT"
    echo "💾 Size: $DATASET_SIZE"
else
    echo "❌ BraTS dataset not found"
    echo "🔄 Check if download is in progress..."
fi

echo ""

# Check GPU status
echo "🖥️  GPU Status:"
echo "==============="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | while read line; do
        echo "  GPU: $line"
    done
else
    echo "⚠️  nvidia-smi not available"
fi

echo ""

# Check system resources
echo "💻 System Resources:"
echo "==================="
echo "🧮 CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "🧠 Memory: $(free -h | awk 'NR==2{printf "%.1f/%.1fGB (%.1f%%)\n", $3/1024/1024, $2/1024/1024, $3*100/$2}')"
echo "💽 Disk: $(df -h . | awk 'NR==2{printf "%s/%s (%s)\n", $3, $2, $5}')"

echo ""

# Check active Python processes
echo "🐍 Active Python Processes:"
echo "==========================="
PYTHON_PROCS=$(ps aux | grep python | grep -v grep | grep -E "(train|kaggle|download)" | head -5)
if [ -n "$PYTHON_PROCS" ]; then
    echo "$PYTHON_PROCS" | while read line; do
        echo "  $line"
    done
else
    echo "  No relevant Python processes found"
fi

echo ""

# Check log files with more details
echo "📄 Log Files Status:"
echo "==================="
for log_file in "pipeline_master.log" "launch_training.log" "fold_0_training.log" "fold_1_training.log" "fold_2_training.log" "training_master.log"; do
    full_path="$LOG_DIR/$log_file"
    if [ -f "$full_path" ]; then
        size=$(stat -c%s "$full_path" 2>/dev/null || echo "0")
        size_human=$(du -sh "$full_path" 2>/dev/null | cut -f1)
        modified=$(stat -c %y "$full_path" 2>/dev/null | cut -d'.' -f1)
        echo "  ✅ $log_file: $size_human ($size bytes)"
        echo "     Last modified: $modified"
    else
        echo "  ❌ $log_file: Not found"
    fi
done

echo ""

# Show pipeline progress
if [ -f "$LOG_DIR/pipeline_master.log" ]; then
    echo "🔄 Pipeline Progress (last 10 lines):"
    echo "====================================="
    tail -10 "$LOG_DIR/pipeline_master.log" 2>/dev/null | sed 's/^/  /'
    echo ""
fi

# Show dataset download progress if in progress
if [ -f "$LOG_DIR/launch_training.log" ]; then
    DOWNLOAD_STATUS=$(grep -i "download\|kaggle\|dataset" "$LOG_DIR/launch_training.log" | tail -3)
    if [ -n "$DOWNLOAD_STATUS" ]; then
        echo "📥 Recent Dataset Activity:"
        echo "=========================="
        echo "$DOWNLOAD_STATUS" | sed 's/^/  /'
        echo ""
    fi
fi

# Show training progress from individual fold logs
for fold in 0 1 2; do
    log_file="$LOG_DIR/fold_${fold}_training.log"
    if [ -f "$log_file" ] && [ -s "$log_file" ]; then
        echo "📈 Fold $fold Progress (last 5 lines):"
        echo "======================================"
        tail -5 "$log_file" 2>/dev/null | sed 's/^/  /'
        echo ""
    fi
done

# Show completion status if available
if [ -f "$LOG_DIR/training_status.txt" ]; then
    echo "🏁 Training Completion Status:"
    echo "============================="
    cat "$LOG_DIR/training_status.txt" | sed 's/^/  /'
    echo ""
fi

echo "🔧 Monitoring Commands:"
echo "======================"
echo "  📊 Real-time pipeline: tail -f $LOG_DIR/pipeline_master.log"
echo "  📥 Dataset download: tail -f $LOG_DIR/launch_training.log"
echo "  🧠 Fold 0 training: tail -f $LOG_DIR/fold_0_training.log"
echo "  🧠 Fold 1 training: tail -f $LOG_DIR/fold_1_training.log"
echo "  🧠 Fold 2 training: tail -f $LOG_DIR/fold_2_training.log"
echo "  🖥️  GPU monitoring: watch -n 1 nvidia-smi"
echo ""
echo "💡 Quick Actions:"
echo "================"
echo "  🔍 Check this status: ./check_training.sh"
echo "  🚀 Launch training: ./launch_training.sh"
echo "  ⏹️  Stop training: kill \$(cat $LOG_DIR/pipeline.pid 2>/dev/null)"