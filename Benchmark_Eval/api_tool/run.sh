#!/usr/bin/env bash
set -euo pipefail

# 1) Input data path
: "${INPUT:?ERROR: please export INPUT path to your jsonl file}"
# Example: export INPUT="/path/to/your/input.jsonl"

# 2) Output directory (one jsonl file per model)
: "${OUT_DIR:?ERROR: please export OUT_DIR path for output}"
# Example: export OUT_DIR="/path/to/output/dir"

mkdir -p "${OUT_DIR}"

# 3) API base URL (use environment variable, do not hardcode)
: "${BASE_URL:?ERROR: please export BASE_URL first}"
# Example: export BASE_URL="http://your-api-server:port/v1/"

# 4) API KEY (do not hardcode)
: "${API_KEY:?ERROR: please export API_KEY first}"

# 5) Model list (run in order)
# Use environment variables or modify to your model names
MODEL_LLAMA="${MODEL_LLAMA:-llama-3.2-11b-vision-instruct}"
MODEL_GPT_5_MINI="${MODEL_GPT_5_MINI:-gpt-5-mini}"
MODEL_GEMINI_2_5_PRO="${MODEL_GEMINI_2_5_PRO:-gemini-2.5-pro-nothinking}"
MODEL_QWEN3_VL_8B="${MODEL_QWEN3_VL_8B:-Qwen/Qwen3-VL-8B-Instruct}"
MODEL_QWEN3_VL_235B="${MODEL_QWEN3_VL_235B:-Qwen/Qwen3-VL-235B-A22B-Instruct}"
MODEL_GEMINI_3_PRO="${MODEL_GEMINI_3_PRO:-gemini-3-pro-preview}"

# 6) Generation parameters (fixed for all models)
TEMPERATURE=0.1
TOP_P=0.95
MAX_TOKENS=16384
TIMEOUT=300  # Large models may need longer timeout

# Default concurrency
CONCURRENCY=8
RETRIES=3

echo "[INFO] BASE_URL=${BASE_URL}"
echo "[INFO] INPUT=${INPUT}"
echo "[INFO] OUT_DIR=${OUT_DIR}"

# =============== Run Model Inference ===============

# Run Llama-3.2-11B-Vision-Instruct
echo ""
echo "==================== RUN MODEL: ${MODEL_LLAMA} ===================="
python run_api_infer.py \
  --input "${INPUT}" \
  --out_dir "${OUT_DIR}" \
  --models "${MODEL_LLAMA}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --max_tokens "${MAX_TOKENS}" \
  --timeout "${TIMEOUT}" \
  --concurrency "${CONCURRENCY}" \
  --retries "${RETRIES}" \
  --prompt_name "parse"

# Run GPT-5-Mini
echo ""
echo "==================== RUN MODEL: ${MODEL_GPT_5_MINI} ===================="
python run_api_infer.py \
  --input "${INPUT}" \
  --out_dir "${OUT_DIR}" \
  --models "${MODEL_GPT_5_MINI}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --max_tokens "${MAX_TOKENS}" \
  --timeout "${TIMEOUT}" \
  --concurrency "${CONCURRENCY}" \
  --retries "${RETRIES}" \
  --prompt_name "parse"

# Run Gemini-2.5-Pro-Nothinking
echo ""
echo "==================== RUN MODEL: ${MODEL_GEMINI_2_5_PRO} ===================="
python run_api_infer.py \
  --input "${INPUT}" \
  --out_dir "${OUT_DIR}" \
  --models "${MODEL_GEMINI_2_5_PRO}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --max_tokens "${MAX_TOKENS}" \
  --timeout "${TIMEOUT}" \
  --concurrency "${CONCURRENCY}" \
  --retries "${RETRIES}" \
  --prompt_name "parse"

# Run Qwen3-VL-8B-Instruct
echo ""
echo "==================== RUN MODEL: ${MODEL_QWEN3_VL_8B} ===================="
python run_api_infer.py \
  --input "${INPUT}" \
  --out_dir "${OUT_DIR}" \
  --models "${MODEL_QWEN3_VL_8B}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --max_tokens "${MAX_TOKENS}" \
  --timeout "${TIMEOUT}" \
  --concurrency "${CONCURRENCY}" \
  --retries "${RETRIES}" \
  --prompt_name "parse"

# Run Qwen3-VL-235B-A22B-Instruct
echo ""
echo "==================== RUN MODEL: ${MODEL_QWEN3_VL_235B} ===================="
python run_api_infer.py \
  --input "${INPUT}" \
  --out_dir "${OUT_DIR}" \
  --models "${MODEL_QWEN3_VL_235B}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --max_tokens "${MAX_TOKENS}" \
  --timeout "${TIMEOUT}" \
  --concurrency "${CONCURRENCY}" \
  --retries "${RETRIES}" \
  --prompt_name "parse"

# Run Gemini-3-Pro-Preview
echo ""
echo "==================== RUN MODEL: ${MODEL_GEMINI_3_PRO} ===================="
python run_api_infer.py \
  --input "${INPUT}" \
  --out_dir "${OUT_DIR}" \
  --models "${MODEL_GEMINI_3_PRO}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --max_tokens "${MAX_TOKENS}" \
  --timeout "${TIMEOUT}" \
  --concurrency "${CONCURRENCY}" \
  --retries "${RETRIES}" \
  --prompt_name "parse"

echo ""
echo "==================== All models completed ===================="
