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
```
pip install git+https://github.com/eigenvivek/xvr.git
```

Verify the installation version (should match the latest release on GitHub):
```
xvr --version
```

## Roadmap

The current repository contains a fully functional package for registering X-ray and CT data. Additionally, pretrained models and data are released such that the results in the paper can be reproduced. 

In the future, extensive documentation, tutorials, and usability improvements (e.g., a user interface) will be added! Feel free to open an issue if there is anything in particular you would like to be added to `xvr`!

- [x] Release a pip-installable version of `xvr`
- [x] Upload pretrained models to reproduce all results in the paper
- [x] Add detailed documentation
- [x] Colab tutorial for iterative pose refinement
- [ ] Colab tutorial for training patient-specific pose regression models
- [ ] User interface for interactive 2D/3D registration

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

#### Models

Pretrained models are available [here](https://huggingface.co/eigenvivek/xvr/tree/main).

#### Data

Benchmarks datasets, reformatted into DICOM/NIfTI files, are available [here](https://huggingface.co/datasets/eigenvivek/xvr-data/tree/main).

If you use the [`DeepFluoro`](https://github.com/rg2/DeepFluoroLabeling-IPCAI2020) dataset, please cite:

    @article{grupp2020automatic,
      title={Automatic annotation of hip anatomy in fluoroscopy for robust and efficient 2D/3D registration},
      author={Grupp, Robert B and Unberath, Mathias and Gao, Cong and Hegeman, Rachel A and Murphy, Ryan J and Alexander, Clayton P and Otake, Yoshito and McArthur, Benjamin A and Armand, Mehran and Taylor, Russell H},
      journal={International journal of computer assisted radiology and surgery},
      volume={15},
      pages={759--769},
      year={2020},
      publisher={Springer}
    }

If you use the [`Ljubljana`](https://lit.fe.uni-lj.si/en/research/resources/3D-2D-GS-CA/) dataset, please cite:

    @article{pernus20133d,
      title={3D-2D registration of cerebral angiograms: A method and evaluation on clinical images},
      author={Mitrović, Uroš and Špiclin, Žiga and Likar, Boštjan and Pernuš, Franjo},
      journal={IEEE transactions on medical imaging},
      volume={32},
      number={8},
      pages={1550--1563},
      year={2013},
      publisher={IEEE}
    }

#### Logging

We use `wandb` to log experiments. To use this feature, set the `WANDB_API_KEY` environment variable by adding the following line to your `.zshrc` or `.bashrc` file:

```zsh
export WANDB_API_KEY=your_api_key
```

## Development

`xvr` is built using [`uv`](https://docs.astral.sh/uv/), an extremely fast Python project manager.

If you want to modify `xvr` (e.g., adding different loss functions, network architectures, etc.), `uv` makes it easy to set up a development environment:

```
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Download xvr
git clone https://github.com/eigenvivek/xvr
cd xvr

# Set up the virtual environment with all dev requirements
uv sync --all-extras
```

To verify your virtual environment, you can run

```
uv run xvr --version
```

Alternatively, you can directly use the virtual environment that `uv` creates:

```
source .venv/bin/activate
xvr --version
```

`xvr`'s [pre-commit hooks](.pre-commit-config.yaml) automatically take care of things like linting and formatting, so hack away! All PRs are welcome.
