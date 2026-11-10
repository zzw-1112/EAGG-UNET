# -*- coding: utf-8 -*-
"""Segmentation metrics: confusion matrix, per-class IoU/Precision/Recall/F1.

Label value 255 is treated as "ignore" and excluded from all statistics.
"""

import numpy as np
import torch


def eval_metrics(output, target, num_classes, conf_matrix):
    """Accumulate segmentation statistics for one batch.

    Args:
        output:      model logits, shape ``[B, C, H, W]``.
        target:      ground-truth labels, shape ``[B, H, W]``.
        num_classes: number of classes ``C``.
        conf_matrix: running confusion matrix ``[C, C]``, updated in place.

    Returns:
        ``(correct, labeled, inter, union, conf_matrix)`` where ``correct`` and
        ``labeled`` are scalars for the current batch, ``inter``/``union`` are
        float arrays of length ``C``.
    """
    _, predicted = torch.max(output, 1)

    # valid pixels only (exclude ignore label 255)
    valid_mask = (target != 255) & (predicted != 255)
    if not valid_mask.any():
        return 0, 0, np.zeros(num_classes), np.zeros(num_classes), conf_matrix

    valid_target = target[valid_mask]
    valid_pred = predicted[valid_mask]

    # drop out-of-range predictions
    range_mask = (valid_target < num_classes) & (valid_pred < num_classes)
    valid_target = valid_target[range_mask]
    valid_pred = valid_pred[range_mask]

    if valid_target.numel() == 0:
        return 0, 0, np.zeros(num_classes), np.zeros(num_classes), conf_matrix

    vt = valid_target.detach().cpu().numpy()
    vp = valid_pred.detach().cpu().numpy()

    correct = int((vt == vp).sum())
    labeled = int(vt.size)

    # local confusion matrix for this batch
    batch_conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(batch_conf, (vt, vp), 1)

    inter = np.diag(batch_conf).astype(np.float64)
    union = (
        batch_conf.sum(axis=1) + batch_conf.sum(axis=0) - np.diag(batch_conf)
    ).astype(np.float64)

    conf_matrix += batch_conf

    return correct, labeled, inter, union, conf_matrix
