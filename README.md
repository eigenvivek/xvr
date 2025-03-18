# `xvr`: X-ray to Volume Registration

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
<a href="https://colab.research.google.com/drive/1K9lBPxcLh55mr8o50Y7aHkjzjEWKPCrM?usp=sharing"><img alt="Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
<a href="https://huggingface.co/eigenvivek/xvr/tree/main" target="_blank"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-ffc107?color=ffc107&logoColor=white"/></a>
<a href="https://huggingface.co/datasets/eigenvivek/xvr-data/tree/main" target="_blank"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data-ffc107?color=ffc107&logoColor=white"/></a>
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

`xvr` is a PyTorch package for differentiable 2D/3D rigid registration with patient-specific learning-based initialization.

This package includes:

- One-line commands for training patient-specific pose regression models from preoperative volumes
- One-line commands for performing iterative pose refinement with different initialization strategies
- A Python API and a CLI

## Installation

Install the Python API and CLI:
```
pip install git+https://github.com/eigenvivek/xvr.git
```

Verify the installation version:
```
xvr --version
```

## Usage

`xvr` provides a command-line interface for training, finetuning, and performing registration (i.e., test-time optimization) with pose regression models. The API is designed to be modular and extensible, allowing users to easily train models on new datasets and anatomical structures without any manual annotations.

```
$ xvr --help

Usage: xvr [OPTIONS] COMMAND [ARGS]...

  xvr is a PyTorch package for training, fine-tuning, and performing 2D/3D
  X-ray to CT/MR registration with pose regression models.

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  train     Train a pose regression model from scratch.
  restart   Restart model training from a checkpoint.
  finetune  Optimize a pose regression model for a specific patient.
  register  Use gradient-based optimization to register XRAY to a CT/MR.
  animate   Animate the trajectory of iterative optimization.
  dcm2nii   Convert a DICOMDIR to a NIfTI file.
```

### Training

To train a pose regression model from scratch on a single patient or a set of preregistered subjects, use `xvr train`:

```
$ xvr train --help

Usage: xvr train [OPTIONS]

  Train a pose regression model from scratch.

Options:
  -i, --inpath PATH              A single CT or a directory of CTs for pretraining  [required]
  -o, --outpath PATH             Directory in which to save model weights  [required]
  --r1 <FLOAT FLOAT>...          Range for primary angle (in degrees)  [required]
  --r2 <FLOAT FLOAT>...          Range for secondary angle (in degrees)  [required]
  --r3 <FLOAT FLOAT>...          Range for tertiary angle (in degrees)  [required]
  --tx <FLOAT FLOAT>...          Range for x-offset (in millimeters)  [required]
  --ty <FLOAT FLOAT>...          Range for y-offset (in millimeters)  [required]
  --tz <FLOAT FLOAT>...          Range for z-offset (in millimeters)  [required]
  --sdd FLOAT                    Source-to-detector distance (in millimeters)  [required]
  --height INTEGER               DRR height (in pixels)  [required]
  --delx FLOAT                   DRR pixel size (in millimeters / pixel)  [required]
  --renderer [siddon|trilinear]  Rendering equation  [default: trilinear]
  --orientation [AP|PA]          Orientation of CT volumes  [default: PA]
  --reverse_x_axis               Enable to obey radiologic convention (e.g., heart on right)
  --parameterization TEXT        Parameterization of SO(3) for regression  [default: euler_angles]
  --convention TEXT              If parameterization is Euler angles, specify order  [default: ZXY]
  --model_name TEXT              Name of model to instantiate  [default: resnet18]
  --pretrained                   Load pretrained ImageNet-1k weights
  --norm_layer TEXT              Normalization layer  [default: groupnorm]
  --lr FLOAT                     Maximum learning rate  [default: 0.005]
  --weight-geo FLOAT             Weight on geodesic loss term  [default: 0.01]
  --batch_size INTEGER           Number of DRRs per batch  [default: 116]
  --n_epochs INTEGER             Number of epochs  [default: 1000]
  --n_batches_per_epoch INTEGER  Number of batches per epoch  [default: 100]
  --name TEXT                    WandB run name
  --project TEXT                 WandB project name  [default: xvr]
  --help                         Show this message and exit.
```

#### Notes
- The `--inpath` argument should point to a directory containing CT volumes for training.
  - If the directory contains a single CT scan, the resulting model be patient-specific.
  - If the directory contains multiple CTs, it's beneficial to preregister them to a common reference frame (e.g., using [ANTs](https://github.com/ANTsX/ANTs)). This will improve the accuracy of the model, but this isn't strictly necessary.

### Finetuning

To finetune a pretrained pose regression model on a new patient, use `xvr finetune`:

```
$ xvr finetune --help

Usage: xvr finetune [OPTIONS]

  Optimize a pose regression model for a specific patient.

Options:
  -i, --inpath PATH              Input CT volume for patient-specific pretraining  [required]
  -o, --outpath PATH             Output directory for finetuned model weights  [required]
  -c, --ckptpath PATH            Checkpoint of a pretrained pose regressor  [required]
  --lr FLOAT                     Maximum learning rate  [default: 0.005]
  --batch_size INTEGER           Number of DRRs per batch  [default: 116]
  --n_epochs INTEGER             Number of epochs  [default: 10]
  --n_batches_per_epoch INTEGER  Number of batches per epoch  [default: 25]
  --rescale FLOAT                Rescale the virtual detector plane  [default: 1.0]
  --name TEXT                    WandB run name
  --project TEXT                 WandB project name  [default: xvr]
  --help                         Show this message and exit.
```

#### Notes

- The `--inpath` argument should point to a single CT volume for which the pose regression model will be finetuned.
- The `--ckpt` argument specifies the path to a checkpoint of a pretrained pose regression model produced by `xvr train`.
  - In addition to model weights, this checkpoint also contains the configurations used for training (e.g., pose parameters and intrinsic parameters), which are reused for finetuning.
 
### Registration (test-time optimization)

To register **real** X-ray images using a pretrained model followed by iterative pose refinement with differentiable rendering, use `xvr register model`:

```
$ xvr register model --help

Usage: xvr register model [OPTIONS] XRAY...

  Initialize from a pose regression model.

Options:
  -v, --volume PATH              Input CT volume (3D image)  [required]
  -m, --mask PATH                Labelmap for the CT volume (optional)
  -c, --ckptpath PATH            Checkpoint of a pretrained pose regressor  [required]
  -o, --outpath PATH             Directory for saving registration results  [required]
  --crop INTEGER                 Preprocessing: center crop the X-ray image  [default: 0]
  --subtract_background          Preprocessing: subtract mode X-ray image intensity
  --linearize                    Preprocessing: convert X-ray from exponential to linear form
  --reducefn TEXT                If DICOM is multiframe, how to extract a single 2D image for registration  [default: max]
  --warp PATH                    SimpleITK transform to warp input CT to template reference frame
  --invert                       Invert the warp
  --labels TEXT                  Labels in mask to exclusively render (comma separated)
  --scales TEXT                  Scales of downsampling for multiscale registration (comma separated)  [default: 8]
  --reverse_x_axis               Enable to obey radiologic convention (e.g., heart on right)
  --renderer [siddon|trilinear]  Rendering equation  [default: trilinear]
  --parameterization TEXT        Parameterization of SO(3) for regression  [default: euler_angles]
  --convention TEXT              If parameterization is Euler angles, specify order  [default: ZXY]
  --lr_rot FLOAT                 Initial step size for rotational parameters  [default: 0.01]
  --lr_xyz FLOAT                 Initial step size for translational parameters  [default: 1.0]
  --patience INTEGER             Number of allowed epochs with no improvement after which the learning rate will be reduced  [default: 10]
  --threshold FLOAT              Threshold for measuring the new optimum  [default: 0.0001]
  --max_n_itrs INTEGER           Maximum number of iterations to run at each scale  [default: 500]
  --max_n_plateaus INTEGER       Number of times loss can plateau before moving to next scale  [default: 3]
  --init_only                    Directly return the initial pose estimate (no iterative pose refinement)
  --saveimg                      Save ground truth X-ray and predicted DRRs
  --pattern TEXT                 Pattern rule for glob is XRAY is directory  [default: *.dcm]
  --verbose INTEGER RANGE        Verbosity level for logging  [default: 1; 0<=x<=3]
  --help                         Show this message and exit.
```

#### Notes

- By passing a `--mask` and a comma-separated set of `--labels`, registration will be performed with respect to specific structures.
- If the model was trained with a coordinate frame different to that of the `--volume`, you can pass a `--warp` to rigidly realign the model's predictions to the new patient.

## Experiments

### Data

### Setup

#### Models



#### Logging

We use `wandb` to log experiments. To use this feature, set the `WANDB_API_KEY` environment variable by adding the following line to your `.zshrc` or `.bashrc` file:

```zsh
export WANDB_API_KEY=your_api_key
```
