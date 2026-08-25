---
icon: lucide/crosshair
---

To register **real** X-ray images with iterative pose refinement with differentiable rendering, use `xvr register`:

- By passing a `--labelpath` and a space-separated set of `--labels`, registration will be performed with respect to specific structures.
- If the model was trained with a coordinate frame different to that of the `--imagepath`, you can pass a `--warp` to rigidly realign the model's predictions to the new patient.

{{ cli("register") }}
