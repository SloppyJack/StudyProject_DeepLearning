import torch


def mixup_batch(images, labels, alpha):
    """Mix a batch and return both targets with their interpolation weight."""
    if alpha <= 0:
        raise ValueError("Mixup alpha must be positive")
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    index = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1 - lam) * images[index]
    return mixed, labels, labels[index], lam
