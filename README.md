# xvr

`xvr` is a command-line interface for training, fine-tuning, and performing 2D/3D registration with pose regression models. The API is designed to be modular and exstensible, allowing users to easily train models on new datasets and anatomical structures without any manual annotations.

```
$ xvr --help

Usage: xvr [OPTIONS] COMMAND [ARGS]...

  xvr is a command-line interface for training, fine-tuning, and performing
  2D/3D X-ray to CT registration with pose regression models.

Options:
  --help  Show this message and exit.

Commands:
  train     Train a pose regression model from scratch.
  restart   Restart model training from a checkpoint.
  finetune  Optimize a pose regression model for a specific patient.
  register  Use gradient-based optimization to register XRAY to a CT.
  animate   Animate the trajectory of iterative optimization.
```
