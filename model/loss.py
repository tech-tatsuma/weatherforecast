import torch
from torchmetrics.regression import LogCoshError


def mse_loss(output, target):
    return torch.nn.MSELoss()(output, target)

def logcosh_loss(output, target):
    loss = LogCoshError().to("cuda")
    return loss(output, target)