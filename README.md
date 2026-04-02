# Scientific Graphics Program Synthesis with Dual Self-Consistency Reinforcement Learning

This repository provides the implementation and evaluation code for SciTikZ, a reinforcement learning framework for synthesizing LaTeX/TikZ code from scientific graphics images. The approach employs a dual self-consistency mechanism to improve both visual fidelity and code quality during training.

## Overview

The core contribution of this work is a dual self-consistency reinforcement learning framework that addresses the challenge of generating accurate LaTeX/TikZ code from scientific diagram images. The framework incorporates two complementary consistency mechanisms:

**Visual Consistency**: The system ensures that the rendered output closely matches the input image by measuring semantic similarity using SIGLIP and structural similarity using LPIPS. This component of the reward function guides the model toward generating code that produces visually accurate outputs.

**Code Consistency**: To promote structural similarity between different code generations for the same input, we employ Token Edit Distance (TED) and CrystalBLEU metrics. This encourages the model to learn more robust and generalizable code patterns.

The overall reward function integrates three components: a binary compilation reward that indicates successful LaTeX compilation, a visual reward based on weighted SIGLIP and LPIPS similarity scores, and a code consistency reward that measures similarity between code variants using TED and CrystalBLEU.

## Project Structure

```
Supplementary Material/
├── EasyR1/                          # RL training framework
│   ├── examples/
│   │   ├── config.yaml              # Training configuration
│   │   ├── qwen3_vl_8b_tikz_visual_grpo.sh  # Training script
│   │   └── reward_function/
│   │       └── tikz_self_consistency.py  # Main reward function
│
├── Data_Process/                     # Data processing pipeline
│   ├── filter.ipynb                 # Data filtering
│   ├── repair/                       # Code repair and distillation
│   └── runtime_validation/           # Batch compilation validation
│
├── Benchmark_Eval/                  # Evaluation framework
│   ├── api_tool/                    # API-based inference
│   └── eval/                         # Evaluation metrics
│
└── LLaMa-Factory/                   # Model training utilities
```

## Installation

### Prerequisites

The code requires Python 3.8 or higher and a CUDA-capable GPU for training. A LaTeX distribution (TeX Live) and ImageMagick are necessary for rendering TikZ code. For containerized training environments, Apptainer or Singularity is required.

### Setup

Install the required Python packages:

```bash
pip install torch torchvision transformers accelerate
pip install lpips pytorch-msssim crystalbleu torchmetrics
pip install pandas pyarrow pillow opencv-python
pip install sacremoses pygments
```

Configure the following environment variables according to your system setup:

```bash
export SIF_PATH="/path/to/apptainer.sif"
export MODEL_PATH="/path/to/model"
export SIGLIP_MODEL_PATH="/path/to/siglip-model"
export TORCH_HOME="/path/to/torch-cache"
export HF_CACHE_ROOT="/path/to/hf-cache"
export TRAIN_PARQUET="/path/to/train.parquet"
export VAL_PARQUET="/path/to/val.parquet"
export SAVE_DIR="/path/to/save/checkpoints"
```

## Usage

### Training

To start reinforcement learning training with the dual self-consistency reward function:

```bash
cd EasyR1/examples
bash qwen3_vl_8b_self_consistancy_rl.sh
```

The training script includes pre-flight checks to verify environment configuration before beginning the training process.

### Evaluation

Run the evaluation pipeline:

```bash
cd Benchmark_Eval/eval

export GT_ROOT="/path/to/ground-truth/images"
export PRED_IMG="/path/to/predicted/images"
export PRED_TEX="/path/to/predicted/code"
export OUT_DIR="/path/to/results"

bash eval.sh
```

### Data Processing

Data filtering can be performed using the provided Jupyter notebook:

```bash
cd Data_Process
jupyter notebook filter.ipynb
```

For batch compilation validation:

```bash
cd runtime_validation
export DISTILL_BASE="/path/to/data"
bash run.sh
```

## Evaluation Metrics

The evaluation framework supports multiple metrics for assessing both image similarity and code similarity. Image similarity is measured using SigLIP, CLIP, LPIPS, SSIM, and DreamSim. Code similarity is evaluated using Token Edit Distance (TED) and CrystalBLEU.

## Acknowledgments

We thank the developers of EasyR1/verl framework, HuggingFace Transformers, and the authors of CrystalBLEU, LPIPS, and SigLIP for providing the foundational tools and metrics used in this work.
