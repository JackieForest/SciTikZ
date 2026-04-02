#!/bin/bash
set -e
set -x
set -o pipefail

# -------------------------
# Image2Code Stage-2 RL (EasyR1/verl) inside apptainer
# -------------------------

# Apptainer SIF file path (required)
: "${SIF_PATH:?ERROR: please export SIF_PATH path to your apptainer.sif file}"
# Example: export SIF_PATH="/path/to/apptainer.sif"
SIF="${SIF_PATH}"

# Optional: Bind host EasyR1 to override container's /opt/EasyR1 (recommended for easy reward/config modifications)
# Set EASYR1_PATH to your EasyR1 directory path, or leave empty to use container's EasyR1
EASYR1_PATH="${EASYR1_PATH:-}"
if [ -n "${EASYR1_PATH}" ]; then
    BIND_EASYR1="--bind ${EASYR1_PATH}:/opt/EasyR1"
else
    BIND_EASYR1=""
fi

# Required binds (can be customized via COMMON_BINDS environment variable)
# Default: only bind /tmp
COMMON_BINDS="${COMMON_BINDS:---bind /tmp:/tmp}"
BINDS="${COMMON_BINDS} ${BIND_EASYR1}"

# -------------------------
# HF / datasets cache: Point to writable disk (critical fix: do not use /workspace)
# -------------------------
: "${HF_CACHE_ROOT:?ERROR: please export HF_CACHE_ROOT path for HuggingFace cache}"
# Example: export HF_CACHE_ROOT="/path/to/hf_cache"
CACHE_ROOT="${HF_CACHE_ROOT}"
mkdir -p "${CACHE_ROOT}/hub" "${CACHE_ROOT}/datasets" "${CACHE_ROOT}/transformers" "${CACHE_ROOT}/xdg"

# -------------------------
# GPU visibility / Ray behavior (critical)
# -------------------------
# 1) --cleanenv will clear CUDA_VISIBLE_DEVICES; explicitly pass it here
# 2) Ray will override (clear) CUDA_VISIBLE_DEVICES on workers with num_gpus=0;
#    Set RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 to disable this behavior (Ray's own suggestion)
CUDA_VISIBLE="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# Runtime cache/temporary directories (all point to writable directories)
ENVVARS="\
--env RAY_TMPDIR=/tmp/ray \
--env TMPDIR=/tmp \
--env HF_HOME=${CACHE_ROOT} \
--env HUGGINGFACE_HUB_CACHE=${CACHE_ROOT}/hub \
--env HF_DATASETS_CACHE=${CACHE_ROOT}/datasets \
--env TRANSFORMERS_CACHE=${CACHE_ROOT}/transformers \
--env XDG_CACHE_HOME=${CACHE_ROOT}/xdg \
--env HF_HUB_DISABLE_TELEMETRY=1 \
--env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE} \
--env RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 \
"

# Training input/output paths
: "${MODEL_PATH:?ERROR: please export MODEL_PATH to your model directory}"
# Example: export MODEL_PATH="/path/to/model"
: "${SAVE_DIR:?ERROR: please export SAVE_DIR path for saving checkpoints}"
# Example: export SAVE_DIR="/path/to/save/dir"
: "${TRAIN_PARQUET:?ERROR: please export TRAIN_PARQUET path to training parquet file}"
# Example: export TRAIN_PARQUET="/path/to/train.parquet"
: "${VAL_PARQUET:?ERROR: please export VAL_PARQUET path to validation parquet file}"
# Example: export VAL_PARQUET="/path/to/val.parquet"

# Python executable inside container
PY=/opt/py312/bin/python

# -------------------------
# (Optional but strongly recommended) Pre-flight checks before training
# -------------------------
apptainer exec --nv --cleanenv --no-home ${ENVVARS} ${BINDS} --pwd /opt/EasyR1 \
  "$SIF" /bin/bash -c "
    set -e
    # Fix PATH to avoid bash -l reading profile and mixing host conda
    export PATH=/opt/py312/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

    echo \"[INFO] python: \$(${PY} -c 'import sys; print(sys.executable)')\"
    ${PY} -V
    echo \"[INFO] CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES\"
    echo \"[INFO] RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=\$RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO\"
    echo \"[INFO] HF_HOME=\$HF_HOME\"

    ${PY} - <<'PY'
import os, pathlib, sys
print('[py]', sys.executable)
print('[env] CUDA_VISIBLE_DEVICES', os.environ.get('CUDA_VISIBLE_DEVICES'))
print('[env] RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO', os.environ.get('RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO'))
hf = os.environ.get('HF_HOME','')
print('[env] HF_HOME', hf)
if hf:
    p = pathlib.Path(hf)
    p.mkdir(parents=True, exist_ok=True)
    t = p / '_write_test'
    t.write_text('ok')
    print('[OK] HF_HOME writable:', t)

mods = ['torch','transformers','vllm','flash_attn','verl','datasets','lpips']
for m in mods:
    try:
        mod = __import__(m)
        print(f'[{m}]', getattr(mod,'__version__','unknown'), getattr(mod,'__file__','unknown'))
    except Exception as e:
        print(f'[{m}] import_failed:', repr(e))

try:
    import torch
    print('[torch] cuda', getattr(torch.version,'cuda',None), 'avail', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('[torch] device_count', torch.cuda.device_count())
        print('[torch] device0', torch.cuda.get_device_name(0))
except Exception as e:
    print('[torch] extra_failed', repr(e))
PY

    command -v pdflatex >/dev/null 2>&1 && pdflatex --version | head -n 1 || (echo '[FATAL] pdflatex not found' && exit 2)
    command -v convert  >/dev/null 2>&1 && convert  --version | head -n 1 || (echo '[FATAL] convert not found'  && exit 3)

    # TikZ smoke test: compile + convert to image, detect rendering pipeline issues early
    work=/tmp/tikz_smoke_\$\$
    mkdir -p \"\$work\"
    cat > \"\$work/main.tex\" <<'TEX'
\\documentclass[tikz]{standalone}
\\usepackage{tikz}
\\begin{document}
\\begin{tikzpicture}
  \\draw (0,0) rectangle (1,1);
\\end{tikzpicture}
\\end{document}
TEX
    (cd \"\$work\" && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null)
    test -f \"\$work/main.pdf\"
    (cd \"\$work\" && convert -density 200 main.pdf\"[0]\" -background white -flatten out.png >/dev/null)
    test -f \"\$work/out.png\"
    echo '[INFO] preflight ok'
  "

# -------------------------
# Training
# -------------------------
# Set reward function paths (use environment variables with defaults)
SIGLIP_MODEL_PATH="${SIGLIP_MODEL_PATH:-/path/to/siglip-model}"
TORCH_HOME="${TORCH_HOME:-/tmp/.torch_cache}"

apptainer exec --nv --cleanenv --no-home ${ENVVARS} ${BINDS} --pwd /opt/EasyR1 \
  "$SIF" ${PY} -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files="${TRAIN_PARQUET}" \
    data.val_files="${VAL_PARQUET}" \
    data.prompt_key=problem \
    data.answer_key=answer \
    data.image_key=images \
    data.image_dir=null \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.model.trust_remote_code=true \
    worker.reward.reward_function=./examples/reward_function/tikz_self_consistency.py:compute_score \
    worker.reward.reward_function_kwargs.siglip_model_path="${SIGLIP_MODEL_PATH}" \
    worker.reward.reward_function_kwargs.lpips_net=alex \
    worker.reward.reward_function_kwargs.device=cuda \
    worker.reward.reward_function_kwargs.torch_home="${TORCH_HOME}" \
    worker.reward.reward_function_kwargs.temp_dir=/tmp/tikz_render \
    worker.reward.reward_function_kwargs.timeout_sec=30 \
    worker.reward.reward_function_kwargs.convert_timeout_sec=30 \
    worker.reward.reward_function_kwargs.density=300 \
    worker.reward.reward_function_kwargs.quality=95 \
    worker.reward.reward_function_kwargs.border=2 \
    worker.reward.reward_function_kwargs.enable_nuclear_no_page=true \
    worker.reward.reward_function_kwargs.size=384 \
    worker.reward.reward_function_kwargs.tau=0.5 \
    worker.reward.reward_function_kwargs.semantic_weight=0.35 \
    worker.reward.reward_function_kwargs.structural_weight=0.65 \
    worker.reward.reward_function_kwargs.compile_success_reward=0.05 \
    worker.reward.reward_function_kwargs.compile_fail_penalty=-0.3 \
    worker.reward.enable_cycle_consistency=true \
    worker.reward.reward_function_kwargs.visual_consistency_threshold=0.7 \
    worker.reward.reward_function_kwargs.code_weight=0.1 \
    worker.reward.reward_function_kwargs.visual_weight=0.85 \
    worker.reward.reward_function_kwargs.code_consistency_threshold=0.6 \
    worker.reward.reward_function_kwargs.code_penalty_max=0.05 \
    worker.reward.reward_function_kwargs.corpus_path="${TRAIN_PARQUET}" \
    worker.reward.reward_function_kwargs.use_apptainer=false \
    worker.reward.reward_function_kwargs.apptainer_bin=/usr/bin/apptainer \
    trainer.project_name=stage2_rl_8b \
    trainer.experiment_name=qwen3_vl_8b_self_consistency_grpo \
    trainer.save_checkpoint_path="${SAVE_DIR}" \
    trainer.n_gpus_per_node=8
