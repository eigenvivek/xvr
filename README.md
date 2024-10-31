# xvr

`xvr` (X-ray to volume registration) is a PyTorch package for differentiable 2D/3D rigid registration.
It provides:

- One line commands for training patient-specific pose regression models from preoperative volumes
- One line commands for performing iterative pose refinement with many different initialization strategies
- A CLI and a Python API

## Installation

```
pip install xvr
```

## Usage

```
$ xvr --help

Usage: xvr [OPTIONS] COMMAND [ARGS]...

  xvr is a PyTorch package for training, fine-tuning, and performing 2D/3D
  X-ray to CT/MR registration with pose regression models.

Options:
  --help  Show this message and exit.

Commands:
  train     Train a pose regression model from scratch.
  restart   Restart model training from a checkpoint.
  finetune  Optimize a pose regression model for a specific patient.
  register  Use gradient-based optimization to register XRAY to a CT/MR.
  animate   Animate the trajectory of iterative optimization.
```