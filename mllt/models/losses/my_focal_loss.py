import torch
import torch.nn as nn
import torch.nn.functional as F

# from ..utils import _log_api_usage_once
from ..builder import LOSSES  # Assuming you're using a registry pattern like MMDetection

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
) -> torch.Tensor:
    if not (0 <= alpha <= 1) and alpha != -1:
        raise ValueError(f"Invalid alpha value: {alpha}. alpha must be in the range [0,1] or -1 for ignore.")
    
    # if not torch.jit.is_scripting() and not torch.jit.is_tracing():
    #     _log_api_usage_once(sigmoid_focal_loss)

    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}. Supported: 'none', 'mean', 'sum'")


@LOSSES.register_module
class MyFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(MyFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        reduction = (
            reduction_override if reduction_override is not None else self.reduction
        )

        loss = sigmoid_focal_loss(
            inputs=cls_score,
            targets=label.float(),
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=reduction,
        )

        return loss * 10
