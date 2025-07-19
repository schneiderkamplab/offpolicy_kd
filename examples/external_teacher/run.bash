#!/bin/bash

# Knowledge Distillation Example Runner
# This script starts the teacher server and then runs the student training

set -e  # Exit on any error

# Script-level configuration variables
TEACHER_MODEL="google/gemma-3-27b-it"
STUDENT_MODEL="google/gemma-3-4b-pt"
TEACHER_DEVICE="cuda:0"  # Separate GPU for teacher (use "cpu" if no GPU available)
SHM_PATH="/tmp/sensai_teacher_shm"
SLOT_NUMBER=0
NUM_SAMPLES=256  # Number of logits to sample per position
MAX_NBYTES=3355443200  # Memory for 256 logits * 1024 positions * 1024 seq_len * 4 bytes * 4 (buffer)
TRAIN_DATA="/path/to/training_data.parquet"  # PLACEHOLDER: Replace with actual training data path
VAL_DATA="/path/to/validation_data.parquet"  # PLACEHOLDER: Replace with actual validation data path

echo "=== SensAI Knowledge Distillation Example ==="
echo "Teacher: $TEACHER_MODEL (on $TEACHER_DEVICE)"
echo "Student: $STUDENT_MODEL (will use remaining GPU/CPU)"
echo "Shared Memory Path: $SHM_PATH"
echo "Slot Number: $SLOT_NUMBER"
echo "Number of samples per position: $NUM_SAMPLES"
echo ""

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Project root: $PROJECT_ROOT"
echo "Script directory: $SCRIPT_DIR"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Basic validation of shared memory path
if [ ! -d "$(dirname "$SHM_PATH")" ]; then
    echo "WARNING: Directory $(dirname "$SHM_PATH") does not exist. Shared memory might not work."
fi

# Check if data files exist
if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: Training data file '$TRAIN_DATA' not found."
    echo "Please update TRAIN_DATA variable with the correct path."
    exit 1
fi

if [ ! -f "$VAL_DATA" ]; then
    echo "ERROR: Validation data file '$VAL_DATA' not found."
    echo "Please update VAL_DATA variable with the correct path."
    exit 1
fi

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ ! -z "$SERVER_PID" ]; then
        echo "Stopping teacher server (PID: $SERVER_PID)"
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi
    
    # Clean up any remaining shared memory files
    rm -f ${SHM_PATH}* 2>/dev/null || true
    
    echo "Cleanup completed"
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

echo "Starting teacher server..."
echo "Command: uv run python -m sensai.logits_server --model $TEACHER_MODEL --device $TEACHER_DEVICE --transport shared_memory --num-clients 1 --shm-path $SHM_PATH --num-samples $NUM_SAMPLES --max-nbytes $MAX_NBYTES --max-tensors 8 --max-dims 8"
echo ""

# Start the teacher server in the background
# Teacher runs on separate device for optimal performance
uv run python -m sensai.logits_server \
    --model $TEACHER_MODEL \
    --device $TEACHER_DEVICE \
    --transport shared_memory \
    --num-clients 1 \
    --shm-path $SHM_PATH \
    --num-samples $NUM_SAMPLES \
    --max-nbytes $MAX_NBYTES \
    --max-tensors 8 \
    --max-dims 8 \
    --interval 0.1 &

SERVER_PID=$!
echo "Teacher server started with PID: $SERVER_PID"

# Wait for server to load (with progress indicator)
echo "Waiting for teacher server to load..."
for i in {1..10}; do
    echo -n "."
    sleep 1
done
echo " done"

# Check if server is still running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "ERROR: Teacher server failed to start or crashed"
    exit 1
fi

echo "Teacher server should be ready now"
echo ""

# Start student training
echo "Starting student training..."
echo "Command: uv run mld-standard --use-external-teacher --sensai-shm-path $SHM_PATH --sensai-slot-number $SLOT_NUMBER --distillation --pretrained --student $STUDENT_MODEL --teacher $TEACHER_MODEL $TRAIN_DATA --val-data-files $VAL_DATA"
echo ""

# Run the student training
uv run mld-standard \
    --use-external-teacher \
    --sensai-shm-path $SHM_PATH \
    --sensai-slot-number $SLOT_NUMBER \
    --distillation \
    --pretrained \
    --student $STUDENT_MODEL \
    --teacher $TEACHER_MODEL \
    $TRAIN_DATA \
    --val-data-files $VAL_DATA

echo ""
echo "Student training completed!"
echo ""
echo "=== Distillation Example Finished ==="