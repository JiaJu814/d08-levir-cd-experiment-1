import torch
import torch.nn.functional as F


def boundary_band(mask, radius=1):
    """方形邻域形态学梯度，边界两侧带宽由radius决定。"""
    dilated = F.max_pool2d(mask, 2 * radius + 1, 1, radius)
    eroded = -F.max_pool2d(-mask, 2 * radius + 1, 1, radius)
    return (dilated - eroded).clamp(0, 1)


def supervised_loss(logits, target, boundary_weight=0.0, radius=1, positive_weight=1.0):
    weights = 1 + boundary_weight * boundary_band(target, radius)
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction='none', pos_weight=logits.new_tensor(positive_weight)) * weights).sum() / weights.sum()
    p = logits.sigmoid()
    dims = (0, 1, 2, 3)
    dice = 1 - ((2 * (p * target).sum(dims) + 1) / (p.sum(dims) + target.sum(dims) + 1)).mean()
    return bce + dice


def pseudo_loss(logits, teacher_probability, high=0.95, low=0.05, boundary_weight=0.0, radius=1, class_balance=False):
    target = (teacher_probability >= 0.5).float()
    keep = ((teacher_probability >= high) | (teacher_probability <= low)).float()
    # 边界损失要求整个边界邻域高置信，避免从未知区域生成错误边界监督。
    safe = -F.max_pool2d(-keep, 2 * radius + 1, 1, radius)
    weights = keep + boundary_weight * boundary_band(target, radius) * safe
    pixel_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    if class_balance:
        pos_weights = weights * target
        neg_weights = weights * (1-target)
        loss = 0.5 * ((pixel_loss*pos_weights).sum()/pos_weights.sum().clamp_min(1) + (pixel_loss*neg_weights).sum()/neg_weights.sum().clamp_min(1))
    else:
        loss = (pixel_loss * weights).sum() / weights.sum().clamp_min(1)
    stats = {'coverage': keep.mean().item(), 'positive_coverage': (keep * target).mean().item()}
    return loss, stats


@torch.no_grad()
def update_teacher(teacher, student, decay):
    for t, s in zip(teacher.parameters(), student.parameters()):
        t.mul_(decay).add_(s, alpha=1 - decay)
    for t, s in zip(teacher.buffers(), student.buffers()):
        if t.is_floating_point():
            t.mul_(decay).add_(s, alpha=1 - decay)
        else:
            t.copy_(s)
