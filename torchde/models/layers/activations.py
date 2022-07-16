import torch


class SwishActivation(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
