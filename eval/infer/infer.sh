# Set working directory (use environment variable or default)
: "${WORK_DIR:?ERROR: please export WORK_DIR path to inference tool directory}"
cd "${WORK_DIR}"

# Set input/output paths (use environment variables)
: "${INPUT_JSONL:?ERROR: please export INPUT_JSONL path to input jsonl file}"
: "${MODEL_PATH:?ERROR: please export MODEL_PATH path to model directory}"
: "${OUTPUT_DIR:?ERROR: please export OUTPUT_DIR path for output}"
: "${PARTITION:?ERROR: please export PARTITION name for Slurm partition}"

# infer script
bash batch_infer.sh \
  --input "${INPUT_JSONL}" \
  --model_path "${MODEL_PATH}" \
  --prompt_name parse \
  --model_name qwen3vl_nothink \
  --parts 1 \
  --gpus 4 \
  --partition "${PARTITION}" \
  --save_images \
  --n_sample 1 \
  --chunk_size 4 \
  --temperature 0.1 \
  --backend hf \
  --repetition_penalty 1.05 \
  --output_dir "${OUTPUT_DIR}" \
  --max_tokens 4096      
