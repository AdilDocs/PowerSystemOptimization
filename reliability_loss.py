import torch
import torch.nn as nn
import torch.nn.functional as F

class ReliabilityLossWithLearnableWeights(nn.Module):
    def __init__(self):
        super().__init__()
        # Raw weights to be normalized by softmax
        self.raw_weights = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))

    def forward(self, pred, target):
        weights = torch.softmax(self.raw_weights, dim=0)
        alpha, beta, gamma = weights[0], weights[1], weights[2]

        loss_lolp = F.mse_loss(pred[:, 0], target[:, 0])
        loss_eens = F.mse_loss(pred[:, 1], target[:, 1])
        loss_lolf = F.mse_loss(pred[:, 2], target[:, 2])

        loss = alpha * loss_lolp + beta * loss_eens + gamma * loss_lolf
        return loss, (alpha.item(), beta.item(), gamma.item())
