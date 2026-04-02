#!/bin/bash
# ==========================================================
# Batch inference script
# Features:
#   1️⃣ Split the input JSONL/HF data into index ranges;
#   2️⃣ Start main.py for each part by srun;
#   3️⃣ Automatically collect logs;
#   4️⃣ Merge results after all tasks are finished.
# ==========================================================

# Activate conda environment (use environment variable or default)
CONDA_ENV="${CONDA_ENV:-rs}"
source activate "${CONDA_ENV}"

# Set working directory (use environment variable or default)
: "${WORK_DIR:?ERROR: please export WORK_DIR path to inference tool directory}"
cd "${WORK_DIR}"
export VLLM_USE_V1=0

# ========== Capture exit signals ==========
cleanup() {
  echo "⚠️ Exit signal captured, terminating all child tasks..."
  pkill -P $$ 2>/dev/null
  echo "🛑 All child tasks have been terminated."
}
trap cleanup EXIT INT TERM

# ========== Default parameters ==========
MODEL_PATH="${MODEL_PATH:-/path/to/model}"
QUOTA_TYPE="reserved"
PROMPT_NAME="parse"
MODEL_NAME="qwen3vl_nothink"
PROMPT_DIR="prompts"
PARTS=4
GPUS=8
PARTITION="${PARTITION:-gpu_partition}"  # Use environment variable or default placeholder
TEMPERATURE=0.1
TOP_P=0.95
MAX_TOKENS=4096
N_SAMPLE=1
CHUNK_SIZE=4
SAVE_IMAGES=${SAVE_IMAGES:-false}  # 🆕 New: whether to save images (default false)
REP_PENALTY=1.0
# ========== Argument parsing ==========
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --input) INPUT="$2"; shift ;;
    --output_dir) OUTPUT_DIR="$2"; shift ;;
    --parts) PARTS="$2"; shift ;;
    --model_path) MODEL_PATH="$2"; shift ;;
    --prompt_name) PROMPT_NAME="$2"; shift ;;
    --model_name) MODEL_NAME="$2"; shift ;;
    --prompt_dir) PROMPT_DIR="$2"; shift ;;
    --partition) PARTITION="$2"; shift ;;
    --gpus) GPUS="$2"; shift ;;
    --save_images) SAVE_IMAGES=true ;;
    --backend) BACKEND="$2"; shift ;;
    --n_sample) N_SAMPLE="$2"; shift ;;                 
    --chunk_size) CHUNK_SIZE="$2"; shift ;;
    --temperature) TEMPERATURE="$2"; shift ;;
    --top_p) TOP_P="$2"; shift ;;
    --max_tokens) MAX_TOKENS="$2"; shift ;;
    --repetition_penalty) REP_PENALTY="$2"; shift ;;
    *) echo "⚠️ Unknown argument: $1"; exit 1 ;;
  esac
  shift
done

# Default backend is vllm
BACKEND=${BACKEND:-vllm}

if [[ -z "$INPUT" ]]; then
  echo "❌ Missing required argument: --input"
  exit 1
fi

# ========== Output directory ==========
BASENAME=$(basename "$INPUT")
EXP_NAME="${BASENAME%.*}"   # Remove extension, e.g. dev.jsonl -> dev
MODEL_PATH_NAME=$(basename "$MODEL_PATH")
MODEL_PATH_NAME=${MODEL_PATH_NAME%.*}

# Output directory is directly under outputs/
OUTPUT_ROOT="${OUTPUT_DIR:-outputs}"   # top-level outputs
LOG_ROOT="logs"

# Experiment subdirectory
EXP_TAG="${EXP_NAME}-${MODEL_NAME}-${PROMPT_NAME}-${MODEL_PATH_NAME}"

OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_TAG}"  # e.g. outputs/dev-qwen2_5vl-parse
LOG_DIR="${LOG_ROOT}/${EXP_TAG}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "📂 Input: $INPUT"
echo "📤 Output directory: $OUTPUT_DIR"
echo "🗒️ Log directory: $LOG_DIR"

# ========== Get input length ==========
echo "🔍 Checking the number of input samples..."
TMP_LEN_FILE="${TMPDIR:-/tmp}/input_len_$$.txt"
python - "$INPUT" > "${TMP_LEN_FILE}" <<'PYCODE'
import json
from datasets import load_dataset
from pathlib import Path
import sys

input_path = sys.argv[1]
try:
    if Path(input_path).exists():
        with open(input_path, 'r', encoding='utf-8') as f:
            n = sum(1 for _ in f)
    else:
        ds = load_dataset(input_path, split='train')
        n = len(ds)
    print(n)
except Exception as e:
    print(-1)
PYCODE

TOTAL=$(cat "${TMP_LEN_FILE}")
rm -f "${TMP_LEN_FILE}"
if [[ "$TOTAL" -le 0 ]]; then
  echo "❌ Unable to determine input length or input is empty (INPUT=$INPUT)."
  exit 1
fi

echo "📊 Total samples: $TOTAL"

# ========== Calculate split ranges ==========
PER_PART=$(( (TOTAL + PARTS - 1) / PARTS ))
echo "🔪 Each task will process about $PER_PART samples"

# ========== Start tasks ==========
echo "🚀 Starting inference tasks..."

for ((i=0; i<PARTS; i++)); do
  START_IDX=$(( i * PER_PART ))
  END_IDX=$(( (i + 1) * PER_PART ))
  if [[ $START_IDX -ge $TOTAL ]]; then
    break
  fi
  if [[ $END_IDX -gt $TOTAL ]]; then
    END_IDX=$TOTAL
  fi

  OUT_FILE="${OUTPUT_DIR}/part_${i}.jsonl"
  LOG_FILE="${LOG_DIR}/part_${i}_$(date +%Y%m%d_%H%M%S).log"

  echo "▶️ Start task part_$i: [$START_IDX, $END_IDX)"
  echo "   Log: $LOG_FILE"

  CMD="python main.py \
      --model_path $MODEL_PATH \
      --input_jsonl $INPUT \
      --output_jsonl $OUT_FILE \
      --prompt_name $PROMPT_NAME \
      --model_name $MODEL_NAME \
      --prompt_dir $PROMPT_DIR \
      --temperature $TEMPERATURE \
      --top_p $TOP_P \
      --max_tokens $MAX_TOKENS \
      --n_sample $N_SAMPLE \
      --chunk_size $CHUNK_SIZE \
      --start_idx $START_IDX \
      --end_idx $END_IDX \
      --backend $BACKEND \
      --repetition_penalty $REP_PENALTY \
      "

  if [[ "$SAVE_IMAGES" == "true" ]]; then
    CMD="$CMD --save_images"   # 🆕 Automatically add parameter
  fi

  srun -p "$PARTITION" --gres=gpu:${GPUS} --async --quotatype=$QUOTA_TYPE bash -c "$CMD" > "$LOG_FILE" 2>&1 &
done

echo "⏳ All tasks have been submitted, waiting for completion..."
wait
echo "✅ All child tasks have finished!"

# ========== Merge outputs ==========
MERGED_FILE="${OUTPUT_DIR}/merged.jsonl"
echo "🧩 Merging results to: $MERGED_FILE"

find "$OUTPUT_DIR" -maxdepth 1 -type f -name "part_*.jsonl" \
  | sort | xargs cat > "$MERGED_FILE"

echo "🎉 All done! Final output file: $MERGED_FILE"
