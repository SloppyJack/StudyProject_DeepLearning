import torch.nn.functional as F


def catdog_aux_loss(logits, labels_a, labels_b, lam):
    """Batch-mean binary loss, weighted by each Mixup target's cat/dog mass.

    Cat=3 maps to 0, dog=5 maps to 1. Other targets contribute zero.
    Dividing by the whole batch avoids amplifying tiny cat/dog mixtures.
    """
    loss = logits.sum() * 0.0
    for labels, weight in ((labels_a, lam), (labels_b, 1 - lam)):
        mask = (labels == 3) | (labels == 5)
        targets = (labels == 5).long()
        per_sample = F.cross_entropy(logits, targets, reduction='none')
        loss = loss + weight * (per_sample * mask).mean()
    return loss
