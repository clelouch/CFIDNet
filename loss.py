import torch
from torch import nn
import torch.nn.functional as F


def _iou(pred, target, size_average=True):
    b = pred.shape[0]
    IoU = 0.0
    for i in range(0, b):
        # compute the IoU of the foreground
        Iand1 = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
        Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - Iand1
        IoU1 = Iand1 / Ior1

        # IoU loss is (1-IoU1)
        IoU = IoU + (1 - IoU1)

    return IoU / b


class IOU(torch.nn.Module):
    def __init__(self, size_average=True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):
        return _iou(pred, target, self.size_average)


def bce_iou_loss(pred, target):
    bce_loss = nn.BCELoss(size_average=True)
    iou_loss_f = IOU(size_average=True)
    bce_out = bce_loss(pred, target)
    iou_out = iou_loss_f(pred, target)

    loss = bce_out + iou_out

    return loss
