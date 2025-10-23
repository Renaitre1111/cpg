# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import torch.nn.functional as F
from torchvision import transforms
from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import IMB_ALGORITHMS
from semilearn.core.utils import get_data_loader
from semilearn.datasets.augmentation import RandAugment
from semilearn.algorithms.hooks import PseudoLabelingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset
from .utils import get_max_confidence_and_residual_variance, batch_class_stats

def get_weights(pred, valid_mask, alpha, epsilon=1e-8):
    weights = torch.zeros_like(pred[:, 0]) # [B]
    max_confidence, residual_variance = get_max_confidence_and_residual_variance(pred)
    if valid_mask.sum() > 0:
        means, vars = batch_class_stats(
            max_confidence[valid_mask],
            residual_variance[valid_mask]
        )
    else:
        means = torch.tensor([1.0, 0.0], device=pred.device)
        vars = torch.tensor([1.0, 1.0], device=pred.device)

    conf_mean = means[0]
    res_mean = means[1]
    conf_var = vars[0]
    res_var = vars[1]

    conf_z = (max_confidence - conf_mean) / torch.sqrt(conf_var + epsilon) # [N]
    res_z = (residual_variance - res_mean) / torch.sqrt(res_var + epsilon) # [N]

    weight_conf = torch.exp(-(conf_z ** 2) / alpha)
    weight_res = torch.exp(-(res_z ** 2) / alpha)

    weight = weight_conf * weight_res # [N]
    confidence_mask = (conf_z > 0) & (res_z > 0) # [N]
    weight = torch.where(confidence_mask, torch.ones_like(weight), weight)

    final_weight = torch.where(valid_mask, weight, torch.zeros_like(weight))

    return final_weight


