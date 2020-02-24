import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(84, 128, 1),
            nn.LeakyReLU(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256,128)
        )

    def forward(self, x):
        #x = x.view(-1, 84)
        shape = x.shape[0]
        x = self.conv(x)
        x = self.fc1(x.view(shape,-1))
        #x = self.fc4(x)
        return F.log_softmax(x, dim=0)
