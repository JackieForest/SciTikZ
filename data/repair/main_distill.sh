#!/bin/bash

# Configuration: set partition name via environment variable
: "${PARTITION:?ERROR: please export PARTITION name for Slurm partition}"
# Example: export PARTITION="your_partition_name"

mkdir -p server_logs
mkdir -p distill_logs
mkdir -p stop_files

# Adjust according to your split count, here is 10 splits: 0~9
for INDEX in 0
do
    bash single_distill.sh $INDEX "${PARTITION}" &
    sleep 1
done

echo "Launched distill loops for INDEX 0"
