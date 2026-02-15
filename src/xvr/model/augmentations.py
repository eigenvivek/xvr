import torch
import torch.nn as nn
import torch.nn.functional as F


def _apply_prob(x, augmented, p, device):
    """Branchless per-sample apply with probability p."""
    B = x.shape[0]
    apply = (torch.rand(B, 1, 1, 1, device=device) < p).to(x.dtype)
    return torch.lerp(x, augmented.to(x.dtype), apply)


class Standardize(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return (x - x.min()) / (x.max() - x.min() + self.eps)


class RandomApproxCLAHE(nn.Module):
    """Approximate CLAHE via adaptive gamma correction."""

    def __init__(self, clip_range=(1.0, 10.0), p=0.5):
        super().__init__()
        self.clip_min, self.clip_max = clip_range
        self.p = p

    def forward(self, x):
        B, C, H, W = x.shape
        clip = torch.empty(B, 1, 1, 1, device=x.device).uniform_(
            self.clip_min, self.clip_max
        )
        mean = x.mean(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        out = x / mean
        out = out.clamp(min=0.0)
        out = torch.min(out, clip) / clip
        out = out * x.amax(dim=(-2, -1), keepdim=True)
        return _apply_prob(x, out, self.p, x.device)


class RandomGamma(nn.Module):
    def __init__(self, gamma_range=(0.7, 1.8), p=0.5):
        super().__init__()
        self.gamma_min, self.gamma_max = gamma_range
        self.p = p

    def forward(self, x):
        B = x.shape[0]
        gamma = torch.empty(B, 1, 1, 1, device=x.device).uniform_(
            self.gamma_min, self.gamma_max
        )
        out = x.clamp(min=1e-6).pow(gamma)
        return _apply_prob(x, out, self.p, x.device)


class RandomBoxBlur(nn.Module):
    def __init__(self, kernel_size=3, p=0.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.p = p

    def forward(self, x):
        C = x.shape[1]
        padded = F.pad(x, [self.pad] * 4, mode="reflect")
        kernel = torch.ones(C, 1, self.kernel_size, self.kernel_size, device=x.device)
        kernel = kernel / (self.kernel_size**2)
        out = F.conv2d(padded, kernel, groups=C)
        return _apply_prob(x, out, self.p, x.device)


class RandomGaussianNoise(nn.Module):
    def __init__(self, std=0.01, p=0.5):
        super().__init__()
        self.std = std
        self.p = p

    def forward(self, x):
        out = x + torch.randn_like(x) * self.std
        return _apply_prob(x, out, self.p, x.device)


class RandomSharpness(nn.Module):
    """Unsharp mask sharpening."""

    def __init__(self, kernel_size=3, p=0.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.p = p

    def forward(self, x):
        C = x.shape[1]
        padded = F.pad(x, [self.pad] * 4, mode="reflect")
        kernel = torch.ones(C, 1, self.kernel_size, self.kernel_size, device=x.device)
        kernel = kernel / (self.kernel_size**2)
        blurred = F.conv2d(padded, kernel, groups=C)
        out = x + (x - blurred)
        return _apply_prob(x, out, self.p, x.device)


class RandomErasing(nn.Module):
    def __init__(self, h_range=(1 / 8, 1 / 3), w_range=(1 / 8, 1 / 3), p=0.5):
        super().__init__()
        self.h_range = h_range
        self.w_range = w_range
        self.p = p

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device

        eh = torch.randint(
            int(H * self.h_range[0]), int(H * self.h_range[1]), (B,), device=device
        )
        ew = torch.randint(
            int(W * self.w_range[0]), int(W * self.w_range[1]), (B,), device=device
        )
        ey = torch.randint(0, H, (B,), device=device)
        ex = torch.randint(0, W, (B,), device=device)

        yy = torch.arange(H, device=device).view(1, H, 1)
        xx = torch.arange(W, device=device).view(1, 1, W)
        mask = (
            (
                (yy >= ey.view(B, 1, 1))
                & (yy < (ey + eh).view(B, 1, 1).clamp(max=H))
                & (xx >= ex.view(B, 1, 1))
                & (xx < (ex + ew).view(B, 1, 1).clamp(max=W))
            )
            .unsqueeze(1)
            .to(x.dtype)
        )

        apply = (torch.rand(B, 1, 1, 1, device=device) < self.p).to(x.dtype)
        mask = mask * apply
        return torch.lerp(x, torch.rand_like(x, dtype=x.dtype), mask)


class RandomCenterCrop(nn.Module):
    """Simulate collimation."""

    def __init__(self, max_crop=10, p=0.5):
        super().__init__()
        self.max_crop = max_crop
        self.p = p

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device

        apply = (torch.rand(B, device=device) < self.p).to(torch.int64)
        crops = torch.randint(0, self.max_crop + 1, (B,), device=device) * apply

        yy = torch.arange(H, device=device).view(1, H, 1)
        xx = torch.arange(W, device=device).view(1, 1, W)
        mask = (
            (yy >= crops.view(B, 1, 1))
            & (yy < H - crops.view(B, 1, 1))
            & (xx >= crops.view(B, 1, 1))
            & (xx < W - crops.view(B, 1, 1))
        ).unsqueeze(1)

        return torch.where(mask, x, torch.zeros_like(x))


class XrayAugmentations(nn.Module):
    def __init__(self, p=0.333, max_crop=10):
        super().__init__()
        self.augmentations = nn.Sequential(
            Standardize(),
            RandomApproxCLAHE(p=p),
            RandomGamma(p=p),
            RandomBoxBlur(p=p),
            RandomGaussianNoise(std=0.01, p=p),
            RandomSharpness(p=p),
            RandomErasing(p=p),
            RandomCenterCrop(max_crop=max_crop, p=p),
        )

    def forward(self, x):
        return self.augmentations(x)
