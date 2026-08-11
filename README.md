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

Usage: xvr [OPTIONS] COMMAND [ARGS]...

  A PyTorch package for 2D/3D XRAY to CT/MR registration.

  Provides functionality for rapidly training pose regression models and
  registering clinical data with gradient-based iterative optimization.

Options:
  -v, --version  Show the version and exit.
  -h, --help     Show this message and exit.

Commands:
  train     Train a pose regression model.
  restart   Restart model training from a checkpoint.
  register  Use gradient-based optimization to register XRAY to a CT/MR.
  animate   Animate the trajectory of iterative optimization.
  dcm2nii   Convert a DICOMDIR to a NIfTI file.
```

### Training

To train a pose regression model from scratch on a single patient or a set of preregistered subjects, use `xvr train`:

```
$ xvr train --help

Usage: xvr train [OPTIONS]

  Train a pose regression model.

Required options:
  -v, --volpath PATH              A single CT or a directory with multiple volumes for pretraining  [required]
  -o, --outpath PATH              Directory in which to save model weights  [required]

Data options:
  -m, --maskpath PATH             Optional labelmaps corresponding to the CTs passed in `volpath`
  --patch_size TEXT               Optional random crop size (e.g., 'h,w,d'); if None, return entire volume
  --num_workers INTEGER           Number of subprocesses to use in the dataloader  [default: 4]
  --pin_memory                    Copy volumes from the dataloader into CUDA pinned memory before returning
  --sample_weights PATH           Probability for sampling each volume in `volpath`

Sampling options:
  --r1 <FLOAT FLOAT>...           Range for primary angle (in degrees)  [required]
  --r2 <FLOAT FLOAT>...           Range for secondary angle (in degrees)  [required]
  --r3 <FLOAT FLOAT>...           Range for tertiary angle (in degrees)  [required]
  --tx <FLOAT FLOAT>...           Range for x-offset (in millimeters)  [required]
  --ty <FLOAT FLOAT>...           Range for y-offset (in millimeters)  [required]
  --tz <FLOAT FLOAT>...           Range for z-offset (in millimeters)  [required]
  --batch_size INTEGER            Number of DRRs per batch  [default: 116]

Renderer options:
  --sdd FLOAT                     Source-to-detector distance (in millimeters)  [required]
  --height INTEGER                DRR height (in pixels)  [required]
  --delx FLOAT                    DRR pixel size (in millimeters / pixel)  [required]
  --renderer [siddon|trilinear]   Rendering equation  [default: trilinear]
  --orientation [AP|PA]           Orientation of CT volumes  [default: AP]
  --reverse_x_axis                Enable to obey radiologic convention (e.g., heart on right)

Model options:
  --model_name TEXT               Name of model to instantiate from the timm library  [default: resnet18]
  --norm_layer TEXT               Normalization layer  [default: groupnorm]
  --pretrained                    Load pretrained ImageNet-1k weights
  --parameterization TEXT         Parameterization of SO(3) for regression  [default: quaternion_adjugate]
  --convention TEXT               If `parameterization='euler_angles'`, specify order  [default: ZXY]
  --unit_conversion_factor FLOAT  Scale factor for translation prediction (e.g., from m to mm)  [default: 1000.0]
  --p_augmentation FLOAT          Base probability of image augmentations during training  [default: 0.333]

Optimizer options:
  --lr FLOAT                      Maximum learning rate  [default: 0.0002]
  --weight_ncc FLOAT              Weight on mNCC loss term  [default: 1.0]
  --weight_geo FLOAT              Weight on geodesic loss term  [default: 0.01]
  --weight_dice FLOAT             Weight on Dice loss term  [default: 1.0]
  --weight_mvc FLOAT              Weight on multiview consistency loss term  [default: 0]
  --n_total_itrs INTEGER          Number of iterations for training the model  [default: 1000000]
  --n_warmup_itrs INTEGER         Number of iterations for warming up the learning rate  [default: 1000]
  --n_grad_accum_itrs INTEGER     Number of iterations for gradient accumulation  [default: 4]
  --n_save_every_itrs INTEGER     Number of iterations before saving a new model checkpoint  [default: 1000]
  --disable_scheduler             Turn off cosine learning rate scheduler

Checkpoint options:
  -c, --ckptpath PATH             Checkpoint of a pretrained pose regressor
  --reuse_optimizer               If ckptpath passed, initialize the previous optimizer's state
  -w, --warp PATH                 SimpleITK transform to warp input CT to the checkpoint's reference frame
  --invert                        Whether to invert the warp or not

Logging options:
  --name TEXT                     WandB run name
  --id TEXT                       WandB run ID (useful when restarting from a checkpoint)
  --project TEXT                  WandB project name  [default: xvr]
```

#### Notes
- The `--volpath` argument should point to a directory containing CT volumes for training.
  - If the directory contains a single CT scan, the resulting model be patient-specific.
  - If the directory contains multiple CTs, it's beneficial to preregister them to a common reference frame (e.g., using [ANTs](https://github.com/ANTsX/ANTs)). This will improve the accuracy of the model, but this isn't strictly necessary.
- We use `wandb` to log experiments. To use this feature, set the `WANDB_API_KEY` environment variable by adding the following line to your `.zshrc` or `.bashrc` file:

    ```bash
    export WANDB_API_KEY=your_api_key
    ```
 
### Registration (test-time optimization)

To register **real** X-ray images using a pretrained model followed by iterative pose refinement with differentiable rendering, use `xvr register model`:

```
$ xvr register model --help

Usage: xvr register model [OPTIONS] XRAY...

  Initialize from a pose regression model.

Required options:
  -c, --ckptpath PATH            Checkpoint of a pretrained pose regressor  [required]
  -v, --volume PATH              Input CT volume (3D image)  [required]
  -o, --outpath PATH             Directory for saving registration results  [required]

Renderer options:
  -m, --mask PATH                Labelmap for the CT volume
  --labels TEXT                  Labels in mask to exclusively render (comma-separated)
  --reverse_x_axis               Enable to obey radiologic convention (e.g., heart on right)
  --renderer [siddon|trilinear]  Renderer equation  [default: trilinear]
  --voxel_shift FLOAT            Position of voxel (top left corner or center)  [default: 0.0]

Preprocessing options:
  --crop INTEGER                 Center crop the X-ray image  [default: 0]
  --subtract_background          Subtract mode X-ray image intensity
  --linearize                    Convert X-ray from exponential to linear form
  --equalize                     Apply histogram equalization to X-rays/DRRs during optimization
  --reducefn TEXT                If DICOM is multiframe, method to extract a single 2D image  [default: max]
  --pattern TEXT                 Pattern rule for glob is XRAY is directory  [default: *.dcm]

Optimizer options:
  --scales TEXT                  Scales of downsampling for multiscale registration (comma-separated)  [default: 8]
  --n_itrs TEXT                  Number of iterations to run at each scale (comma-separated)  [default: 500]
  --parameterization TEXT        Parameterization of SO(3) for regression  [default: euler_angles]
  --convention TEXT              If parameterization is Euler angles, specify order  [default: ZXY]
  --lr_rot FLOAT                 Initial step size for rotational parameters  [default: 0.01]
  --lr_xyz FLOAT                 Initial step size for translational parameters  [default: 1.0]
  --patience INTEGER             Number of itrs without improvement before decreasing the learning rate  [default: 10]
  --threshold FLOAT              Threshold for measuring the new optimum  [default: 0.0001]
  --max_n_plateaus INTEGER       Number of times loss can plateau before moving to next scale  [default: 3]

Logging options:
  --init_only                    Directly return the initial pose estimate (no iterative pose refinement)
  --saveimg                      Save ground truth X-ray and predicted DRRs
  --verbose INTEGER RANGE        Verbosity level for logging  [default: 1; 0<=x<=3]

Miscellaneous options:
  --warp PATH                    SimpleITK transform to warp input CT to a template reference frame
  --invert                       Whether to invert the warp or not
  --antipodal                    Initialize from antipode of predicted pose
```

#### Notes

- By passing a `--mask` and a comma-separated set of `--labels`, registration will be performed with respect to specific structures.
- If the model was trained with a coordinate frame different to that of the `--volume`, you can pass a `--warp` to rigidly realign the model's predictions to the new patient.
