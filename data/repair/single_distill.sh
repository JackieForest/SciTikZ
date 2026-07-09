#!/bin/bash
export INDEX=$1
export CLUSTER=$2

# Configuration: set paths via environment variables
: "${VLLM_SIF_PATH:?ERROR: please export VLLM_SIF_PATH}"
: "${MODEL_PATH:?ERROR: please export MODEL_PATH}"
: "${WORK_DIR:?ERROR: please export WORK_DIR}"
: "${EXCLUDE_NODES:-}"  # Optional: exclude specific nodes
: "${BIND_PATHS:-}"  # Optional: apptainer bind paths (e.g., "/share:/share,/mnt:/mnt")

# Example:
# export VLLM_SIF_PATH="/path/to/vllm-cu128.sif"
# export MODEL_PATH="/path/to/model"
# export WORK_DIR="/path/to/work/dir"
# export EXCLUDE_NODES="node1,node2,node3"  # Optional
# export BIND_PATHS="/share:/share,/mnt:/mnt"  # Optional: adjust according to your cluster

STOP_FILE="stop_files/distill_${INDEX}.flag"

while true
do
    if [ -f "$STOP_FILE" ]; then
        echo "Stop flag detected for INDEX=${INDEX}! Stopping job submission..."
        break
    fi

    EXCLUDE_OPT=""
    if [ -n "$EXCLUDE_NODES" ]; then
        EXCLUDE_OPT="-x $EXCLUDE_NODES"
    fi

    srun -p $CLUSTER --gres=gpu:8 --quotatype=reserved --job-name=${INDEX}_write --time=1440 \
        $EXCLUDE_OPT \
    bash -c '
        echo "Starting distill job with INDEX=$INDEX"

        # Current node list
        nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")

        # Extract head node IP (adjust PREFIX based on your cluster naming convention)
        # This is a generic approach - modify the sed pattern to match your node naming
        head_ip=$(echo "$nodes" | head -1 | sed "s/^[^0-9]*//" | tr "-" ".")
        echo "Head node IP: $head_ip"

        # Start vLLM service
        BIND_OPT=""
        if [ -n "$BIND_PATHS" ]; then
            BIND_OPT="--bind $BIND_PATHS"
        fi
        apptainer exec --nv $BIND_OPT \
            "$VLLM_SIF_PATH" \
            vllm serve "$MODEL_PATH" \
            --max-model-len 64000 --enable-expert-parallel --tensor-parallel-size 8 \
            --async-scheduling --limit-mm-per-prompt.video 0 \
            --gpu-memory-utilization 0.96 --enable-chunked-prefill \
            --max-num-seqs 96 --enforce-eager \
            2>&1 | tee server_logs/${head_ip}.log &

        echo "Waiting for vLLM service to start..."

        TIMEOUT=600
        ELAPSED=0
        INTERVAL=20

        while [ $ELAPSED -lt $TIMEOUT ]; do
            if grep -q "Application startup complete." server_logs/${head_ip}.log 2>/dev/null; then
                echo "vLLM service started successfully!"
                break
            fi
            sleep $INTERVAL
            ELAPSED=$((ELAPSED + INTERVAL))
            echo "Waited ${ELAPSED}s..."
        done

        if [ $ELAPSED -lt $TIMEOUT ]; then
            # Ensure running local_api.py in the work directory
            cd "$WORK_DIR"
            python local_api.py --index $INDEX --url $head_ip 2>&1 | tee "distill_logs/${head_ip}.log"
        else
            echo "vLLM service failed to start within TIMEOUT."
        fi
    '
    sleep 10
done
