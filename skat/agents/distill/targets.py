import torch


def visit_targets(visits, masks):
    counts = visits.to(torch.float32).masked_fill(~masks, 0)
    totals = counts.sum(-1, keepdim=True)
    if (counts < 0).any() or (totals <= 0).any():
        raise ValueError("each training row requires positive legal visits")
    return counts / totals
