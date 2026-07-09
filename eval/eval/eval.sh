#!/usr/bin/env bash
set -euo pipefail

# ====== Manually load conda (do not rely on /etc/profile or ~/.bashrc) ======
# Modify according to your actual conda installation location: common locations are ~/miniconda3 or ~/anaconda3
if [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  echo "[FATAL] conda.sh not found under \$HOME/anaconda3 or \$HOME/miniconda3" >&2
  exit 1
fi

# Set conda environment name via environment variable
CONDA_ENV="${CONDA_ENV:-env_name}"
conda activate "${CONDA_ENV}"

# ====== Path Configuration ======
GT_ROOT="/path/to/ground/truth/root"
GT_IMG="${GT_ROOT}/images"
GT_TEX="${GT_ROOT}/codes"

PRED_IMG="/path/to/predicted/images"
PRED_TEX="/path/to/predicted/tex"

OUT_DIR="/path/to/evaluation/output"

# ====== Model Paths (modify according to your local setup) ======
SIGLIP_MODEL="/path/to/siglip/model/siglip-so400m-patch14-384"
CLIP_MODEL="/path/to/clip/model/openai_clip-vit-large-patch14"

# ====== DreamSim Configuration ======
DREAMSIM_MODEL="ensemble"

# ====== Dependencies (only needed when packages are missing) ======
# pip install -U transformers pillow torch torchvision pytorch-msssim lpips pandas tqdm dreamsim huggingface_hub

mkdir -p "${OUT_DIR}"

# ====== Execute Evaluation ======
python evaluate.py \
  --gt_img_dir "${GT_IMG}" \
  --gt_tex_dir "${GT_TEX}" \
  --pred_img_dir "${PRED_IMG}" \
  --pred_tex_dir "${PRED_TEX}" \
  --out_dir "${OUT_DIR}" \
  --siglip_model "${SIGLIP_MODEL}" \
  --clip_model "${CLIP_MODEL}" \
  --dreamsim_model "${DREAMSIM_MODEL}" \
  --device cuda \
  --batch_size 16 \
  --lpips_net alex \
  --lpips_tau 0.5 \
  --ssim_resize 384 \
  --trivial_top_k 500
