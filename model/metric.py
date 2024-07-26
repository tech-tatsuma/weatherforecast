import torch


def mse_loss(output, target):
    return torch.nn.MSELoss()(output, target)