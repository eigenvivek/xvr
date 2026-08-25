# `xvr`: X-ray to Volume Registration

[![docs](https://github.com/eigenvivek/xvr/actions/workflows/docs.yml/badge.svg)](https://github.com/eigenvivek/xvr/actions/workflows/docs.yml)
[![Paper shield](https://img.shields.io/badge/arXiv-2503.16309-red.svg)](https://arxiv.org/abs/2503.16309)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
<a href="https://colab.research.google.com/drive/1K9lBPxcLh55mr8o50Y7aHkjzjEWKPCrM?usp=sharing"><img alt="Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
<a href="https://huggingface.co/eigenvivek/xvr/tree/main" target="_blank"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-ffc107?color=ffc107&logoColor=white"/></a>
<a href="https://huggingface.co/datasets/eigenvivek/xvr-data/tree/main" target="_blank"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data-ffc107?color=ffc107&logoColor=white"/></a>
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

A PyTorch package for training patient-specific 2D/3D registration models **in 5 minutes.**

<p align="center">
  <img width="410" alt="image" src="https://github.com/user-attachments/assets/8a01c184-f6f1-420e-82b9-1cbe733adf7f" />
</p>

## Highlights

- 🚀 A single CLI/API for training models and registering clinical data
- ⚡️ **100x faster** patient-specific model training than [`DiffPose`](https://github.com/eigenvivek/DiffPose)
- 📐 Submillimeter registration accuracy with new image-similarity metrics
- 🩺 Human-interpretable pose parameters for **training your own models**
- 🐍 Pure Python/PyTorch implementation
- 🖥️ Supports macOS, Linux, and Windows

`xvr` is built upon [`DiffDRR`](https://github.com/eigenvivek/DiffDRR), the differentiable X-ray renderer.

## Installation

Install the Python API and CLI (should take ~5 min if installing PyTorch with CUDA):
```bash
pip install git+https://github.com/eigenvivek/xvr.git
```

Verify the installation version (should match the latest release on GitHub):
```bash
xvr --version
```

## CLI Usage

`xvr` provides a command-line interface for training/finetuning pose regression models and registering clinical data with gradient-based iterative optimization with trained models. It is designed to be modular and extensible, allowing users to easily train models on new datasets and anatomical structures without any manual annotations. Full documentation is available [here](https://xvr.csail.mit.edu/).

```
$ xvr --help

Usage: xvr COMMAND

Commands:
register     Use gradient-based optimization to register XRAY to a CT/MR.
restart      Restart model training from a checkpoint.
train        Train a pose regression model.
--help -h    Display this message and exit.
--version    Display application version.
```

## Development

`xvr` is built using [`uv`](https://docs.astral.sh/uv/), an extremely fast Python project manager.

If you want to modify `xvr` (e.g., adding different loss functions, network architectures, etc.), `uv` makes it easy to set up a development environment:

```bash
# Download xvr
git clone https://github.com/eigenvivek/xvr && cd xvr

# Install uv and build the environment with all dev requirements
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --all-groups

# Install pre-commit hooks locally
uv tool install prek
uvx prek install -f
```

To verify your virtual environment, you can run

```bash
uv run xvr --version
```

Alternatively, you can directly use the virtual environment that `uv` creates:

```bash
source .venv/bin/activate
xvr --version
```

`xvr`'s [pre-commit hooks](.pre-commit-config.yaml) automatically take care of things like linting and formatting, so hack away! All PRs are welcome.

## Experiments

Reproducing the paper's registration results requires a CUDA GPU. First, build the environment with `uv`:

```bash
git clone https://github.com/eigenvivek/xvr.git && cd xvr
uv sync --all-groups
```

Then download the [pretrained models](https://huggingface.co/eigenvivek/xvr) (3.7 GB) and [datasets](https://huggingface.co/datasets/eigenvivek/xvr-data) (4.8 GB) from HuggingFace:

```bash
uvx hf download eigenvivek/xvr      --repo-type model   --local-dir experiments/models/
uvx hf download eigenvivek/xvr-data --repo-type dataset --local-dir experiments/data/
```

Registration runs three datasets (DeepFluoro, Femur, Ljubljana) × three initializations (_de novo_, finetuned, foundation) as nine SLURM array jobs:

```bash
./experiments/run.sh register
```

The scripts are in `experiments/scripts/{dataset}/register/`. Four `#SBATCH` directives are cluster-specific: update `--partition`, `--qos`, `--account`, and `--gres` to match your platform. Metrics were computed on an NVIDIA RTX 6000 Ada with PyTorch 2.10.

If you don't have SLURM, you can run the subjects in series by manually supplying the array index:

```bash
for m in de_novo finetuned foundation; do
    for i in $(seq 1 6);  do SLURM_ARRAY_TASK_ID=$i bash experiments/scripts/deepfluoro/register/$m.sh; done
    for i in $(seq 1 5);  do SLURM_ARRAY_TASK_ID=$i bash experiments/scripts/femur/register/$m.sh;      done
    for i in $(seq 1 10); do SLURM_ARRAY_TASK_ID=$i bash experiments/scripts/ljubljana/register/$m.sh;  done
done
```

Once every job has finished, score the results:

```bash
./experiments/run.sh evaluate
```

This writes `experiments/results/registration.csv`, rebuilt from scratch on each run, with one row per x-ray per pose (`init` and `final`) recording mPE, mRPE, mTRE, dGeo, the final NCC, and runtime.
