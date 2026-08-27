import torch

from xvr.model.loss import DiceLoss, DiceMetric, HausdorffLoss


def make_labelmap(*channels) -> torch.Tensor:
    """Stack 2D channels into (1, C, H, W); channel 0 is background by convention."""
    return torch.stack([torch.as_tensor(c, dtype=torch.float32) for c in channels])[None]


def test_dice_of_identical_masks_is_one():
    mask = make_labelmap(torch.zeros(4, 4), torch.eye(4))
    torch.testing.assert_close(DiceMetric()(mask, mask), torch.ones(1, 1))


def test_dice_of_disjoint_masks_is_zero():
    a = torch.zeros(4, 4)
    a[0] = 1.0
    b = torch.zeros(4, 4)
    b[1] = 1.0
    metric = DiceMetric()(make_labelmap(torch.zeros(4, 4), a), make_labelmap(torch.zeros(4, 4), b))
    torch.testing.assert_close(metric, torch.zeros(1, 1))


def test_dice_matches_the_closed_form_on_a_partial_overlap():
    """2|A n B| / (|A| + |B|): 2*2/(4+4) = 0.5."""
    a = torch.zeros(4, 4)
    a[0, :4] = 1.0
    b = torch.zeros(4, 4)
    b[0, 2:] = 1.0
    b[1, :2] = 1.0
    metric = DiceMetric()(make_labelmap(torch.zeros(4, 4), a), make_labelmap(torch.zeros(4, 4), b))
    torch.testing.assert_close(metric, torch.full((1, 1), 0.5))


def test_dice_excludes_the_background_channel():
    """The metric returns `dice[:, 1:]`, so a C-channel input yields C-1 scores."""
    mask = make_labelmap(torch.ones(4, 4), torch.eye(4), torch.zeros(4, 4))
    assert DiceMetric()(mask, mask).shape == (1, 2)


def test_dice_loss_is_the_complement_and_survives_empty_channels():
    """An empty channel gives 0/0; `nanmean` then `nan_to_num` must keep the loss finite."""
    empty = make_labelmap(torch.zeros(4, 4), torch.eye(4), torch.zeros(4, 4))
    loss = DiceLoss()(empty, empty)
    assert torch.isfinite(loss).all()
    torch.testing.assert_close(loss, torch.zeros(1))


def test_hausdorff_of_identical_masks_is_zero():
    """The loss weights a squared difference, so identical inputs cost nothing."""
    mask = make_labelmap(torch.zeros(8, 8), torch.eye(8))
    torch.testing.assert_close(HausdorffLoss()(mask, mask), torch.zeros(1))


def test_hausdorff_penalizes_a_displaced_mask():
    a = torch.zeros(8, 8)
    a[1:3, 1:3] = 1.0
    b = torch.zeros(8, 8)
    b[5:7, 5:7] = 1.0
    loss = HausdorffLoss()(make_labelmap(torch.zeros(8, 8), a), make_labelmap(torch.zeros(8, 8), b))
    assert loss.item() > 0.0


def test_hausdorff_handles_a_labelmap_with_only_background():
    """Dropping channel 0 can leave nothing behind; the `C == 0` guard covers that."""
    only_background = torch.zeros(1, 1, 8, 8)
    torch.testing.assert_close(HausdorffLoss()(only_background, only_background), torch.zeros(1))


def test_hausdorff_is_differentiable_through_the_prediction():
    """The distance transform is detached, but the squared difference must carry gradients."""
    truth = make_labelmap(torch.zeros(8, 8), torch.eye(8))
    pred = make_labelmap(torch.zeros(8, 8), torch.rand(8, 8)).requires_grad_(True)
    HausdorffLoss()(truth, pred).sum().backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
