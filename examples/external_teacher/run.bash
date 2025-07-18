#!/bin/bash

# Knowledge Distillation Example Runner
# This script starts the teacher server and then runs the student training

set -e  # Exit on any error

# Script-level configuration variables
TEACHER_MODEL="HuggingFaceTB/SmolLM2-135M"
STUDENT_MODEL="HuggingFaceTB/SmolLM2-135M"
SHM_PATH="/tmp/sensai_teacher_shm"
SLOT_NUMBER=0
NUM_SAMPLES=256  # Number of logits to sample per position
TRAIN_DATA="train_data.parquet"
VAL_DATA="val_data.parquet"

echo "=== SensAI Knowledge Distillation Example ==="
echo "Teacher: $TEACHER_MODEL"
echo "Student: $STUDENT_MODEL"
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
echo "Command: uv run python -m sensai.logits_server --model $TEACHER_MODEL --device cpu --transport shared_memory --num-clients 1 --shm-path $SHM_PATH --num-samples $NUM_SAMPLES --max-nbytes 3000000000 --max-tensors 8 --max-dims 8"
echo ""

# Start the teacher server in the background
uv run python -m sensai.logits_server \
    --model $TEACHER_MODEL \
    --device cpu \
    --transport shared_memory \
    --num-clients 1 \
    --shm-path $SHM_PATH \
    --num-samples $NUM_SAMPLES \
    --max-nbytes 3000000000 \
    --max-tensors 8 \
    --max-dims 8 \
    --interval 0.1 &

SERVER_PID=$!
echo "Teacher server started with PID: $SERVER_PID"

# Wait for server to load
echo "Waiting 10 seconds for teacher server to load..."
sleep 10

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