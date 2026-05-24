import torch
import torch.nn as nn
import torch.nn.functional as F


class InterFrameDifferenceLoss(nn.Module):
    """
    基于相邻帧差分的一致性损失（时序一致性）。

    支持两种输入形状：
    - (B, T, H, W)
    - (B, T, C, H, W)

    参数：
    - p: 1 为 L1，2 为 L2（MSE）；默认 1
    - reduction: 'none' | 'mean' | 'sum'
    """

    def __init__(self, p: int = 1, reduction: str = 'mean') -> None:
        super().__init__()
        assert p in (1, 2), "p 只支持 1 (L1) 或 2 (L2/MSE)"
        assert reduction in ('none', 'mean', 'sum'), "reduction 必须是 'none' | 'mean' | 'sum'"
        self.p = p
        self.reduction = reduction

    def forward(self, reconstructed_sequence: torch.Tensor, ground_truth_sequence: torch.Tensor) -> torch.Tensor:
        # 形状校正为 (B, T, C, H, W)
        def ensure_5d(x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 4:
                # (B, T, H, W) -> (B, T, 1, H, W)
                return x.unsqueeze(2)
            if x.dim() == 5:
                return x
            raise ValueError("输入张量必须是 (B,T,H,W) 或 (B,T,C,H,W) 形状")

        rec = ensure_5d(reconstructed_sequence)
        gt = ensure_5d(ground_truth_sequence)

        if rec.size(1) < 2:
            return torch.tensor(0.0, device=rec.device, dtype=rec.dtype)

        rec_diff = rec[:, 1:, ...] - rec[:, :-1, ...]
        gt_diff = gt[:, 1:, ...] - gt[:, :-1, ...]

        if self.p == 1:
            return F.l1_loss(rec_diff, gt_diff, reduction=self.reduction)
        # p == 2
        return F.mse_loss(rec_diff, gt_diff, reduction=self.reduction)


def interframe_difference_loss(
    reconstructed_sequence: torch.Tensor,
    ground_truth_sequence: torch.Tensor,
    p: int = 1,
    reduction: str = 'mean',
) -> torch.Tensor:
    """函数式接口，等价于 InterFrameDifferenceLoss 的 forward。

    Args:
        reconstructed_sequence: (B,T,H,W) 或 (B,T,C,H,W)
        ground_truth_sequence: (B,T,H,W) 或 (B,T,C,H,W)
        p: 1 (L1) 或 2 (L2/MSE)
        reduction: 'none' | 'mean' | 'sum'
    """
    return InterFrameDifferenceLoss(p=p, reduction=reduction)(reconstructed_sequence, ground_truth_sequence)