import torch
import torch.nn as nn

class SimAM(nn.Module):
    def __init__(self, lambda_=0.0001):
        """
        SimAM attention module.
        :param lambda_: A coefficient lambda in the equation.
        """
        super(SimAM, self).__init__()
        self.lambda_ = lambda_

    def forward(self, X):
        """
        Forward pass for SimAM.
        :param X: Input feature map of shape (N, C, H, W)
        :return: Output feature map with attention applied, same shape as input (N, C, H, W)
        """
        # Calculate the spatial size minus 1 for normalization
        n = X.shape[2] * X.shape[3] - 1

        # Calculate the square of (X - mean(X))
        d = (X - X.mean(dim=[2, 3], keepdim=True)).pow(2)

        # Channel variance (d.sum() / n)
        v = d.sum(dim=[2, 3], keepdim=True) / n

        # Calculate E_inv which contains importance of X
        E_inv = d / (4 * (v + self.lambda_)) + 0.5

        # Return attended features using sigmoid activation
        return X * torch.sigmoid(E_inv)
