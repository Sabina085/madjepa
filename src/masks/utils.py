# Code adapted from: https://github.com/facebookresearch/ijepa

import torch


def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep -> List([B, K]); K = number of patches kept by one mask in the list
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        # after unsqueeze(-1) -> [B, K, 1]
        # after repeat -> [B, K, D]
        all_x += [torch.gather(x, dim=1, index=mask_keep)]

    # if you have M masks in the list -> output = [M*B, K, D]
    return torch.cat(all_x, dim=0)
