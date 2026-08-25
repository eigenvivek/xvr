# `xvr`: X-ray to Volume Registration

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

You can also enable tab-completion for `xvr` by adding this line to your `~/.bashrc` (instructions for other shells are [here](https://click.palletsprojects.com/en/stable/shell-completion/)):

```bash
eval "$(_XVR_COMPLETE=bash_source xvr)"
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

The scripts are in `experiments/scripts/{dataset}/register/`. Four `#SBATCH` directives are cluster-specific: update `--partition`, `--qos`, `--account`, and `--gres` to match your platform. Note that the reported metrics were computed on an NVIDIA RTX 6000 Ada with PyTorch 2.10.

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

## CLI Usage

`xvr` provides a command-line interface for training/finetuning pose regression models and registering clinical data with gradient-based iterative optimization with trained models. The API is designed to be modular and extensible, allowing users to easily train models on new datasets and anatomical structures without any manual annotations.

```
$ xvr --help

Usage: xvr COMMAND

Commands:
register   Use gradient-based optimization to register XRAY to a CT/MR.
restart    Restart model training from a checkpoint.
train      Train a pose regression model.
--help -h  Display this message and exit.
--version  Display application version.
```

### Training

To train a pose regression model from scratch on a single patient or a set of preregistered subjects, use `xvr train`:

```
$ xvr train --help

Usage: xvr train [OPTIONS]

Train a pose regression model.

Data:
* --volpath -v TEXT              CT or directory of CTs for pretraining [required]
* --outpath -o TEXT              Directory in which to save model weights [required]
  --maskpath -m TEXT             Optional labelmaps corresponding to the CTs
  --patch-size TEXT              Optional random crop size e.g. 'h,w,d'; if None,
                                 return entire volume
  --sample-weights TEXT          Probability for sampling each volume in volpath
  --num-workers INTEGER          Number of subprocesses to use in the dataloader
                                 [default: 4]
  --pin-memory --no-pin-memory   Copy volumes into CUDA pinned memory before
                                 returning [default: False]

Sampling:
* --r1 <FLOAT FLOAT>...          Range for primary angle (in degrees) [required]
* --r2 <FLOAT FLOAT>...          Range for secondary angle (in degrees) [required]
* --r3 <FLOAT FLOAT>...          Range for tertiary angle (in degrees) [required]
* --tx <FLOAT FLOAT>...          Range for x-offset (in millimeters) [required]
* --ty <FLOAT FLOAT>...          Range for y-offset (in millimeters) [required]
* --tz <FLOAT FLOAT>...          Range for z-offset (in millimeters) [required]
  --batch-size INTEGER           Number of DRRs per batch [default: 116]
  --img-threshold FLOAT          Minimum fraction of foreground pixels to keep a DRR
                                 [default: 0.1]
  --mask-threshold FLOAT         Minimum fraction of mask pixels to keep a DRR
                                 [default: 0.05]
  --n-samples INTEGER            Number of points sampled along each ray when
                                 rendering DRRs [default: 500]
  --geodesic-only                Skip re-rendering from predicted poses; supervise
    --no-geodesic-only           pose directly with geodesic loss only (fast)
                                 [default: False]

Renderer:
* --sdd FLOAT                    Source-to-detector distance (in millimeters)
                                 [required]
* --height INTEGER               DRR height (in pixels) [required]
* --delx FLOAT                   DRR pixel size (in millimeters / pixel) [required]
  --orientation TEXT             Orientation of CT volumes [choices: AP, PA]
                                 [default: AP]
  --reverse-x-axis               Obey radiologic convention (e.g., heart on right)
    --no-reverse-x-axis          [default: False]

Model:
  --model-name TEXT              Name of model to instantiate from the timm library
                                 [default: resnet18]
  --norm-layer TEXT              Normalization layer [default: groupnorm]
  --pretrained --no-pretrained   Load pretrained ImageNet-1k weights [default:
                                 False]
  --parameterization TEXT        Parameterization of SO(3) for regression [default:
                                 quaternion_adjugate]
  --convention TEXT              If parameterization='euler_angles', specify order
                                 [default: ZXY]
  --unit-conversion-factor FLOAT Scale factor for translation prediction (e.g., from
                                 m to mm) [default: 1000.0]
  --p-augmentation FLOAT         Base probability of image augmentations during
                                 training [default: 0.333]

Optimizer:
  --lr FLOAT                     Maximum learning rate [default: 0.0002]
  --weight-ncc FLOAT             Weight on mNCC loss term [default: 1.0]
  --weight-geo FLOAT             Weight on geodesic loss term [default: 0.01]
  --weight-dice FLOAT            Weight on Dice loss term [default: 1.0]
  --weight-haus FLOAT            Weight on Hausdorff loss term [default: 0.1]
  --n-total-itrs INTEGER         Number of iterations for training the model
                                 [default: 1000000]
  --n-warmup-itrs INTEGER        Number of iterations for warming up the learning
                                 rate [default: 1000]
  --n-grad-accum-itrs INTEGER    Number of iterations for gradient accumulation
                                 [default: 4]
  --n-save-every-itrs INTEGER    Number of iterations before saving a new model
                                 checkpoint [default: 1000]
  --disable-scheduler            Turn off cosine learning rate scheduler [default:
    --no-disable-scheduler       False]

Checkpoint:
  --ckptpath TEXT                Checkpoint of a pretrained pose regressor
  --reuse-optimizer              Initialize the previous optimizer's state [default:
    --no-reuse-optimizer         False]
  --warp TEXT                    SimpleITK transform to warp input CT to
                                 checkpoint's reference frame
  --invert --no-invert           Whether to invert the warp or not [default: False]

Logging:
  --project TEXT                 WandB project name [default: xvr]
  --group TEXT                   WandB run group
  --name TEXT                    WandB run name
  --id TEXT                      WandB run ID (useful when restarting from a
                                 checkpoint)
```

#### Notes
- The `--volpath` argument should point to a directory containing CT volumes for training.
  - If the directory contains a single CT scan, the resulting model be patient-specific.
  - If the directory contains multiple CTs, it's beneficial to preregister them to a common reference frame (e.g., using [Greedy](https://greedy.readthedocs.io/en/latest/install.html)). This will improve the accuracy of the model, but this isn't strictly necessary.
- We use `wandb` to log experiments. To use this feature, set the `WANDB_API_KEY` environment variable by adding the following line to your `.zshrc` or `.bashrc` file:

    ```bash
    export WANDB_API_KEY=your_api_key
    ```
 
### Registration (test-time optimization)

To register **real** X-ray images using a pretrained model followed by iterative pose refinement with differentiable rendering, use `xvr register model`:

```
$ xvr register model --help

Usage: xvr register model --files LIST[PATH] --imagepath STR [OPTIONS] CKPT

Register using a neural network initial pose estimate.

Parameters:
* --files PATH...                X-ray images to register [required]

MODEL:
* CKPT --ckpt TEXT               Path to model checkpoint [required]
  --warp TEXT                    SimpleITK transform reframing a model's predicted
                                 pose
  --antipodal --no-antipodal     Initialize from the antipode of the predicted pose
                                 [default: False]

Data:
* --imagepath TEXT               Path to the CT image [required]
  --labelpath TEXT               Path to the segmentation label map. If None, uses
                                 the full image
  --labels INTEGER...            Label indices to include in the DRR. If None, uses
                                 all labels

Optimizer:
  --metric TEXT                  Image similarity metric [choices: mncc, gncc,
                                 gmncc] [default: gmncc]
  --scales FLOAT...              Downsampling scale(s) for multiscale registration
                                 [default: [8.0]]
  --n-itrs INTEGER...            Number of optimization iterations per scale
                                 [default: [500]]
  --lr-rot FLOAT                 Learning rate for rotation parameters [default:
                                 0.01]
  --lr-xyz FLOAT                 Learning rate for translation parameters [default:
                                 1.0]
  --lr-reduce-factor FLOAT       Factor by which to reduce the learning rate on
                                 plateau [default: 0.1]
  --patience INTEGER...          Number of steps with no improvement before reducing
                                 the learning rate (one per scale) [default: [5]]
  --threshold FLOAT              Minimum change to qualify as an improvement
                                 [default: 0.0001]
  --max-n-plateaus INTEGER       Number of learning rate reductions before early
                                 stopping [default: 2]
  --parameterization TEXT        Parameterization of SO(3) for pose optimization
                                 [default: euler_angles]
  --convention TEXT              If parameterization='euler_angles', specify order
                                 [default: ZXY]
  --init-only --no-init-only     Return initial pose estimate result [default:
                                 False]

Preprocessing:
  --crop INTEGER                 Number of pixels to crop from the image border
                                 [default: 0]
  --linearize --no-linearize     Convert image to linear attenuation values
                                 [default: True]
  --subtract-background          Subtract background from the image [default: False]
    --no-subtract-background
  --equalize --no-equalize       Apply histogram equalization during optimization
                                 [default: False]
  --reducefn TEXT                Reduction function for multi-frame images [choices:
                                 max, sum] [default: max]

Miscellaneous:
  --device TEXT                  Torch device to run on [default: cuda]
  --savepath TEXT                Location to save the registration results
  --saveplot --no-saveplot       Save plots of registration results [default: False]
```

#### Notes

- By passing a `--labelpath` and a space-separated set of `--labels`, registration will be performed with respect to specific structures.
- If the model was trained with a coordinate frame different to that of the `--imagepath`, you can pass a `--warp` to rigidly realign the model's predictions to the new patient.
