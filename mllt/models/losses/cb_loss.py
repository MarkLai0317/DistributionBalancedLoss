"""Pytorch implementation of Class-Balanced-Loss
   Reference: "Class-Balanced Loss Based on Effective Number of Samples" 
   Authors: Yin Cui and
               Menglin Jia and
               Tsung Yi Lin and
               Yang Song and
               Serge J. Belongie
   https://arxiv.org/abs/1901.05555, CVPR'19.
"""


import numpy as np
import torch
import torch.nn.functional as F

import torch.nn as nn

import mmcv

from ..registry import LOSSES



# 

def focal_loss(labels, logits, alpha, gamma):
    """
    Compute focal loss for multi-label classification with mean normalization.

    Args:
        labels: Tensor of shape [batch_size, num_classes], with binary labels (0 or 1).
        logits: Tensor of shape [batch_size, num_classes].
        alpha: Tensor of shape [batch_size, num_classes], per-class weight for each sample.
        gamma: Scalar, focal modulation factor.

    Returns:
        Scalar focal loss (mean over all elements).
    """
    BCE_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')  # [B, C]

    if gamma == 0.0:
        modulator = 1.0
    else:
        prob = torch.sigmoid(logits)
        p_t = prob * labels + (1 - prob) * (1 - labels)  # [B, C]
        modulator = (1 - p_t) ** gamma  # [B, C]

    loss = modulator * BCE_loss  # [B, C]
    weighted_loss = alpha * loss  # [B, C]

    focal_loss = weighted_loss.mean()  # mean over all elements
    return focal_loss * 5




def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.

    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.

    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.

    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    

    # weights = torch.tensor(weights).float()
    # weights = weights.unsqueeze(0)
    # weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
    # weights = weights.sum(1)
    # weights = weights.unsqueeze(1)
    # weights = weights.repeat(1,no_of_classes)


    weights = torch.tensor(weights).float().to(logits.device)  # shape: [C]
    weights = weights.unsqueeze(0)  # shape: [1, C]
    weights = weights.repeat(labels.shape[0], 1)  # shape: [B, C]
    # print("label_size", labels.shape)
    # print("logits_size", logits.shape)


    # weights = torch.tensor(weights).float().to(logits.device)  # shape: [C]
    # weights = weights.unsqueeze(0)  # shape: [1, C]
    # weights = weights.repeat(labels.shape[0],1) * labels
    # weights = weights.sum(1)
    # weights = weights.unsqueeze(1)
    # weights = weights.repeat(1,no_of_classes)
   

    if loss_type == "focal":
        cb_loss = focal_loss(labels, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels.float(), weight = weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim = 1)
        cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
    # print("cb_loss", cb_loss)
    return cb_loss



# Register module if you have a registry system
@LOSSES.register_module
class CBLoss(nn.Module):
    def __init__(self,
                 freq_file,
                 no_of_classes,
                 loss_type='focal',
                 beta=0.999,
                 gamma=2.0):
        super(CBLoss, self).__init__()
        self.samples_per_cls = torch.from_numpy(np.asarray(
            mmcv.load(freq_file)['class_freq'])).float()
        # self.samples_per_cls = samples_per_cls
        self.no_of_classes = no_of_classes
        self.loss_type = loss_type
        self.beta = beta
        self.gamma = gamma

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        # label is expected as class index (not one-hot)
        # if label.shape == cls_score.shape:
        #     label = label.argmax(dim=1)

        return CB_loss(
            labels=label,
            logits=cls_score,
            samples_per_cls=self.samples_per_cls,
            no_of_classes=self.no_of_classes,
            loss_type=self.loss_type,
            beta=self.beta,
            gamma=self.gamma
        )



if __name__ == '__main__':
    no_of_classes = 5
    logits = torch.rand(10,no_of_classes).float()
    labels = torch.randint(0,no_of_classes, size = (10,))
    beta = 0.9999
    gamma = 2.0
    samples_per_cls = [2,3,1,2,2]
    loss_type = "focal"
    cb_loss = CB_loss(labels, logits, samples_per_cls, no_of_classes,loss_type, beta, gamma)
    print(cb_loss)