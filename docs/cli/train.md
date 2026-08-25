---
icon: lucide/brain
---

To train a pose regression model from scratch on a single patient or a set of preregistered subjects, use `xvr train`:

- The `--volpath` argument should point to a directory containing CT volumes for training.
  - If the directory contains a single CT scan, the resulting model be patient-specific.
  - If the directory contains multiple CTs, it's beneficial to preregister them to a common reference frame (e.g., using [Greedy](https://greedy.readthedocs.io/en/latest/install.html)). This will improve the accuracy of the model, but this isn't strictly necessary.
- We use `wandb` to log experiments. To use this feature, set the `WANDB_API_KEY` environment variable by adding the following line to your `.zshrc` or `.bashrc` file:

    ```bash
    export WANDB_API_KEY=your_api_key
    ```

{{ cli("train") }}
