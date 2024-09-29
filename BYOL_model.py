import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class BYOL_model(nn.Module):
    def __init__(self):
        super(BYOL_model, self).__init__()

        self.resnet = torchvision.models.resnet18()

    def forward(self, x):
        return self.resnet(x)