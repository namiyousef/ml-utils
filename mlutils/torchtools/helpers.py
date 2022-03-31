import torch
def unique_first(x):
    """ Torch function to get the first appearance unique items"""
    unique, inverse = torch.unique(x, sorted=True, return_inverse=True)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
    return perm