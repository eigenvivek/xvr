import torch
import torch.nn.functional as F
from jaxtyping import Float


def mind_ssc_2d(
    img: Float[torch.Tensor, "B 1 H W"],
    radius: int = 2,
    dilation: int = 2,
) -> Float[torch.Tensor, "B 4 H W"]:
    """Computes the 2D MIND-SSC descriptor.

    Produces a dense 4-channel descriptor that characterizes local structure
    through pairwise patch comparisons within a 4-connected neighborhood. The
    descriptor is modality-independent, making it suitable for multi-modal
    image registration.

    Reference:
        Heinrich et al., "Towards Realtime Multimodal Fusion for Image-Guided
        Interventions Using Self-similarities", MICCAI 2013.

    Args:
        img: Input image tensor.
        radius: Radius for patch averaging. The patch size used for computing
            local squared differences is ``(2 * radius + 1) ** 2``.
        dilation: Spacing between the center voxel and its neighbors. Controls
            the spatial extent of the self-similarity context.

    Returns:
        MIND-SSC descriptor with 4 channels, one per neighbor pair. Values are
        in (0, 1], where 1 indicates maximum self-similarity.
    """
    device = img.device
    kernel_size = 2 * radius + 1

    # 4-connected neighborhood of center [1, 1] in a 3x3 grid:
    #   [0, 1] = up, [1, 0] = left, [1, 2] = right, [2, 1] = down
    neighbors = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]], device=device)

    # Select pairs at squared distance 2 (sqrt(2) apart), excluding opposite
    # pairs (up-down, left-right) which have squared distance 4. Upper triangle
    # avoids duplicates, yielding 4 pairs.
    diff = neighbors.unsqueeze(1) - neighbors.unsqueeze(0)
    sq_dist = (diff**2).sum(-1)
    i, j = torch.meshgrid(
        torch.arange(4, device=device), torch.arange(4, device=device), indexing="ij"
    )
    mask = ((i > j) & (sq_dist == 2)).view(-1)

    idx_a = neighbors.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 2)[mask]
    idx_b = neighbors.unsqueeze(0).expand(4, -1, -1).reshape(-1, 2)[mask]
    num_pairs = idx_a.shape[0]

    # Build 3x3 shift kernels. Each kernel contains a single 1 at one neighbor
    # position, so convolving with it shifts the image to that location.
    def _build_shift_kernels(
        idx: Float[torch.Tensor, "num_pairs 2"],
    ) -> Float[torch.Tensor, "num_pairs 1 3 3"]:
        kernels = torch.zeros(num_pairs, 1, 3, 3, device=device)
        flat_idx = torch.arange(num_pairs, device=device) * 9 + idx[:, 0] * 3 + idx[:, 1]
        kernels.view(-1)[flat_idx] = 1
        return kernels

    shift_a = _build_shift_kernels(idx_a)
    shift_b = _build_shift_kernels(idx_b)

    # Compute patch-based mean squared differences between each neighbor pair.
    img_padded = F.pad(img, [dilation] * 4, mode="replicate")
    sample_a = F.conv2d(img_padded, shift_a, dilation=dilation)
    sample_b = F.conv2d(img_padded, shift_b, dilation=dilation)

    patch_msd = F.avg_pool2d(
        F.pad((sample_a - sample_b) ** 2, [radius] * 4, mode="replicate"),
        kernel_size,
        stride=1,
    )

    # Normalize: subtract per-pixel minimum, divide by mean, exponentiate.
    descriptor = patch_msd - patch_msd.min(dim=1, keepdim=True).values
    variance = descriptor.mean(dim=1, keepdim=True)
    variance = variance.clamp(
        min=variance.mean().item() * 1e-3,
        max=variance.mean().item() * 1e3,
    )
    descriptor = torch.exp(-descriptor / variance)

    return descriptor
