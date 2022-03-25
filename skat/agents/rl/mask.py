import torch


class MaskedCategorical(torch.distributions.categorical.Categorical):
    def __init__(self, logits: torch.Tensor, mask: torch.Tensor):
        self.mask = mask
        self.mask_value = torch.finfo(logits.dtype).min  # -∞
        logits.masked_fill_(~self.mask, self.mask_value)
        super().__init__(logits=logits)
